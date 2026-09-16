"""Leaf masks for every image, via SAM with a centre-point prompt.

Strategy, per spec section 3:

1. Prompt SAM with a single positive point at the image centre and keep the
   highest-scoring of its three candidate masks.
2. If SAM's predicted IoU is below ``score_threshold``, or the mask is
   degenerate (almost empty / almost the whole frame), fall back to an
   HSV-green largest-connected-component segmentation.
3. Cache the result as a binary PNG mirroring the raw tree.

The run is **resumable**: an image whose mask PNG already exists is skipped, so
an interrupted pass can simply be re-run. Every decision is recorded in a
per-dataset CSV so the fallback rate is visible rather than silent — if SAM is
quietly failing on half the dataset you need to know before training on it.

Usage::

    python masks/leaf_masks.py --config configs/leaf_masks_plantvillage.yaml
    python masks/leaf_masks.py --config configs/leaf_masks_plantdoc.yaml --mobile_sam
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, load_config, set_seed  # noqa: E402
from data.dataset import split_raw_path  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

# OpenCV hue is 0-179. Healthy and diseased leaf tissue both sit in this band;
# it is deliberately wide because the fallback only needs to be roughly right.
HSV_GREEN_LOW = np.array([20, 30, 25], dtype=np.uint8)
HSV_GREEN_HIGH = np.array([95, 255, 255], dtype=np.uint8)

MaskMethod = Literal["sam", "hsv_fallback", "unreliable"]


def hsv_leaf_mask(image_rgb: np.ndarray) -> np.ndarray:
    """Green connected component containing the image centre, as ``{0,1}`` uint8.

    Used when SAM is unconfident. **Prefers the centre component, not the
    largest.** On a field photograph the largest green blob is usually the
    surrounding vegetation, so "largest" selects the whole scene; the subject
    leaf is the one under the centre of the frame. Falls back to largest only
    when the centre is not green.

    Returns zeros when no green region is found. Callers must treat an empty or
    near-total result as unreliable rather than using it - see
    :func:`assess_reliability`.
    """
    h, w = image_rgb.shape[:2]
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    raw = cv2.inRange(hsv, HSV_GREEN_LOW, HSV_GREEN_HIGH)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel, iterations=1)
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((raw > 0).astype(np.uint8), 8)
    if n <= 1:
        return np.zeros((h, w), dtype=np.uint8)

    centre_label = int(labels[h // 2, w // 2])
    if centre_label > 0:
        chosen = centre_label
    else:
        chosen = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    mask = (labels == chosen).astype(np.uint8)

    # Fill interior holes so lesions inside the leaf stay part of the leaf.
    filled = mask.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, contours, -1, color=1, thickness=cv2.FILLED)
    return filled


def assess_reliability(mask: np.ndarray, lo: float = 0.02, hi: float = 0.85) -> bool:
    """Whether a finished leaf mask is usable.

    A mask covering more than ``hi`` of the frame is not isolating a leaf, it is
    selecting the scene. Measured on PlantDoc, HSV-fallback masks had a median
    coverage of 0.82 and an upper quartile of 0.95 - i.e. most of them were
    useless, while looking like successes in the output directory. Downstream
    consumers (counterfactual cutouts, bgremoval at eval) must skip these rather
    than build on them.
    """
    cov = float(mask.mean())
    return lo <= cov <= hi


def _is_degenerate(mask: np.ndarray, lo: float = 0.02, hi: float = 0.98) -> bool:
    """True if the mask covers almost nothing or almost everything."""
    cov = float(mask.mean())
    return cov < lo or cov > hi


class LeafMasker:
    """Wraps SAM (or MobileSAM) with an HSV fallback.

    Args:
        checkpoint: Path to the SAM checkpoint.
        model_type: SAM registry key, ``vit_b`` per spec.
        device: Torch device.
        mobile_sam: Use MobileSAM instead. The spec requires this option below
            8 GB VRAM; this machine has 6 GB, so it is the safety valve if
            ViT-B OOMs.
        score_threshold: Minimum SAM predicted IoU to accept its mask.
    """

    def __init__(
        self,
        checkpoint: Path,
        model_type: str = "vit_b",
        device: torch.device | None = None,
        mobile_sam: bool = False,
        score_threshold: float = 0.85,
    ):
        self.device = device or get_device()
        self.score_threshold = score_threshold
        self.mobile_sam = mobile_sam
        self.predictor = None

        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint}\n"
                "Download it with:\n"
                "  curl -L -o weights/sam_vit_b_01ec64.pth "
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            )

        if mobile_sam:
            try:
                from mobile_sam import SamPredictor, sam_model_registry  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "mobile_sam is not installed. Either install it "
                    "(pip install git+https://github.com/ChaoningZhang/MobileSAM.git) "
                    "or drop --mobile_sam to use SAM ViT-B."
                ) from exc
        else:
            from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)

    def mask_one(self, image_rgb: np.ndarray) -> tuple[np.ndarray, float, MaskMethod, bool]:
        """Return ``(mask, score, method, reliable)`` for one RGB image.

        The mask is ``{0,1}`` uint8 at the image's own resolution. ``reliable``
        is False when neither SAM nor the fallback produced a plausible leaf;
        the mask is still written so the run stays resumable, but downstream
        code must skip it.
        """
        h, w = image_rgb.shape[:2]
        try:
            self.predictor.set_image(image_rgb)
            point = np.array([[w // 2, h // 2]])
            label = np.array([1])
            masks, scores, _ = self.predictor.predict(
                point_coords=point, point_labels=label, multimask_output=True
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

        # Prefer the best-scoring candidate, but if it is degenerate try the
        # other two before giving up on SAM entirely - the 3 candidates are
        # coarse/medium/fine, and the coarse one is often the whole scene.
        order = np.argsort(-scores)
        for idx in order:
            cand = masks[int(idx)].astype(np.uint8)
            if assess_reliability(cand) and float(scores[int(idx)]) >= self.score_threshold:
                return cand, float(scores[int(idx)]), "sam", True

        best = int(order[0])
        sam_mask = masks[best].astype(np.uint8)
        score = float(scores[best])

        fallback = hsv_leaf_mask(image_rgb)
        if assess_reliability(fallback):
            return fallback, score, "hsv_fallback", True

        # Neither worked. Keep SAM's mask if it is at least non-degenerate,
        # otherwise the fallback, and flag the image as unusable.
        chosen = sam_mask if not _is_degenerate(sam_mask) else fallback
        return chosen, score, "unreliable", False


def iter_images(root: Path, classes: set[str] | None = None) -> list[Path]:
    """Image files under ``root``, sorted for a deterministic (resumable) order.

    ``classes`` restricts to those immediate parent folder names. Useful for
    generating one crop's masks first when the full set is hours of work and a
    downstream step only needs that crop.
    """
    if not root.exists():
        raise FileNotFoundError(f"Image root not found: {root}")
    out = [p for p in sorted(root.rglob("*")) if p.suffix in IMAGE_EXTS and p.is_file()]
    if classes is not None:
        out = [p for p in out if p.parent.name in classes]
        if not out:
            raise ValueError(f"No images under {root} matched classes={sorted(classes)}")
    return out


def classes_for_crop(class_map_path: Path, crop: str, column: str) -> set[str]:
    """Dataset folder names for one crop, from ``class_map.csv``."""
    import pandas as pd

    df = pd.read_csv(class_map_path)
    sel = df[(df.crop.str.lower() == crop.lower()) & df[column].notna()]
    return set(sel[column].astype(str))


def run(cfg: dict, repo: Path, mobile_sam_override: bool | None = None) -> dict[str, int]:
    """Generate leaf masks for one dataset. Returns per-method counts."""
    image_root = repo / cfg["image_root"]
    mask_root = repo / cfg.get("mask_root", "data/masks/leaf")
    checkpoint = repo / cfg.get("checkpoint", "weights/sam_vit_b_01ec64.pth")
    threshold = float(cfg.get("score_threshold", 0.85))
    limit = cfg.get("limit")
    mobile = mobile_sam_override if mobile_sam_override is not None else cfg.get("mobile_sam", False)

    classes = None
    if cfg.get("crop"):
        classes = classes_for_crop(
            repo / cfg.get("class_map", "data/class_map.csv"),
            str(cfg["crop"]),
            cfg.get("class_map_column", "plantvillage_name"),
        )
        print(f"crop filter {cfg['crop']!r}: {len(classes)} classes", flush=True)

    images = iter_images(image_root, classes)
    if limit:
        images = images[: int(limit)]
    print(f"found {len(images)} images under {image_root}", flush=True)

    # Resume: skip anything already on disk before loading the model at all.
    todo = []
    for p in images:
        dataset, relative = split_raw_path(p)
        out = mask_root / dataset / f"{relative.as_posix()}.png"
        if out.exists() and out.stat().st_size > 0:
            continue
        todo.append((p, out))
    print(f"{len(images) - len(todo)} already done, {len(todo)} to generate", flush=True)
    if not todo:
        return {"sam": 0, "hsv_fallback": 0, "skipped": len(images)}

    masker = LeafMasker(
        checkpoint=checkpoint,
        model_type=cfg.get("model_type", "vit_b"),
        mobile_sam=bool(mobile),
        score_threshold=threshold,
    )
    print(f"model ready on {masker.device} (mobile_sam={bool(mobile)})", flush=True)

    dataset_name, _ = split_raw_path(todo[0][0])
    log_path = mask_root / dataset_name / "_leaf_mask_log.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()

    counts = {"sam": 0, "hsv_fallback": 0}
    t0 = time.time()

    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["relative_path", "method", "sam_score", "coverage", "reliable"])

        for i, (src, out) in enumerate(todo):
            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Could not read image: {src}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            mask, score, method, reliable = masker.mask_one(rgb)
            counts[method] = counts.get(method, 0) + 1

            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), mask * 255)

            _, relative = split_raw_path(src)
            writer.writerow(
                [
                    relative.as_posix(),
                    method,
                    round(score, 4),
                    round(float(mask.mean()), 4),
                    int(reliable),
                ]
            )

            if i % 200 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                eta = (len(todo) - i - 1) / max(rate, 1e-6) / 60
                fb = counts.get("hsv_fallback", 0) / max(i + 1, 1)
                bad = counts.get("unreliable", 0) / max(i + 1, 1)
                print(
                    f"  {i + 1}/{len(todo)}  {rate:.1f} img/s  eta {eta:.1f} min  "
                    f"fallback {fb:.1%}  unreliable {bad:.1%}",
                    flush=True,
                )
                fh.flush()

    total = sum(counts.values())
    bad = counts.get("unreliable", 0)
    print(
        f"done: {total} masks in {(time.time() - t0) / 60:.1f} min | "
        f"sam={counts.get('sam', 0)} hsv_fallback={counts.get('hsv_fallback', 0)} "
        f"unreliable={bad} "
        f"({counts.get('hsv_fallback', 0) / max(total, 1):.1%} fallback, "
        f"{bad / max(total, 1):.1%} unreliable)",
        flush=True,
    )
    if bad:
        print(
            f"WARNING: {bad} image(s) have no usable leaf mask. They are written but flagged "
            "reliable=0 in the log; downstream code must skip them rather than build on them.",
            flush=True,
        )
    print(f"log: {log_path}", flush=True)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate leaf masks with SAM.")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--mobile_sam", action="store_true", help="use MobileSAM (for GPUs below 8 GB VRAM)"
    )
    ap.add_argument("--limit", type=int, default=None, help="process only the first N images")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    if args.limit is not None:
        cfg["limit"] = args.limit
    set_seed(int(cfg.get("seed", 0)))

    run(cfg, repo, mobile_sam_override=True if args.mobile_sam else None)


if __name__ == "__main__":
    main()
