"""Run the S0 segmenter over PlantVillage to produce pseudo lesion masks (Plan A).

Output goes to ``data/masks/lesion_pseudo/`` and is **training-only**. Reported
metrics may never be computed on it — ``eval`` refuses pseudo masks without an
explicit ``--allow_pseudo``. Masks are promoted into ``lesion_human/`` only after
a person checks them, via ``verify_sheet.py`` then ``import_verified.py``.

Predictions are clipped to inside the SAM leaf mask. A field-trained segmenter
applied to lab images will occasionally fire on background texture, and an
uncorrected lesion mask containing background would invert the very thing the
CAM penalty is meant to teach.

Usage::

    python masks/pseudo_label.py --config configs/pseudo_label_tomato.yaml
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, load_config, set_seed  # noqa: E402
from data.dataset import IMAGENET_MEAN, IMAGENET_STD, mask_path_for, split_raw_path  # noqa: E402
from masks.train_segmenter import build_segmenter  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def load_segmenter(ckpt_path: Path, device: torch.device):
    """Load the S0 checkpoint and return ``(model, img_size)``."""
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Segmenter checkpoint not found: {ckpt_path}\n"
            "Train it first: python masks/train_segmenter.py --config configs/S0_segmenter.yaml"
        )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_segmenter(ck.get("arch", "unet"), ck.get("encoder", "resnet34"))
    model.load_state_dict(ck["model_state"])
    model.to(device).eval()
    print(
        f"loaded segmenter: arch={ck.get('arch')} crop={ck.get('crop')} "
        f"val_iou={ck.get('val_iou'):.4f} @ epoch {ck.get('epoch')}",
        flush=True,
    )
    return model, int(ck.get("img_size", 512))


def tomato_classes(class_map_path: Path, crop: str = "tomato") -> set[str]:
    """PlantVillage folder names for one crop, from ``class_map.csv``."""
    df = pd.read_csv(class_map_path)
    sel = df[(df.crop.str.lower() == crop.lower()) & df.plantvillage_name.notna()]
    return set(sel.plantvillage_name.astype(str))


@torch.no_grad()
def predict_probs(model, image_rgb: np.ndarray, size: int, device: torch.device) -> np.ndarray:
    """Foreground probability map at the image's own resolution."""
    h, w = image_rgb.shape[:2]
    resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.autocast("cuda", enabled=device.type == "cuda"):
        logits = model(t)
    probs = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    return cv2.resize(probs, (w, h), interpolation=cv2.INTER_LINEAR)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate pseudo lesion masks with the S0 segmenter.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    device = get_device()

    image_root = repo / cfg["image_root"]
    leaf_root = repo / cfg.get("leaf_mask_root", "data/masks/leaf")
    out_root = repo / cfg.get("out_root", "data/masks/lesion_pseudo")
    ckpt = repo / cfg.get("checkpoint", "weights/lesion_seg.pt")
    threshold = float(cfg.get("threshold", 0.5))
    crop = cfg.get("crop", "tomato")

    keep = tomato_classes(repo / cfg.get("class_map", "data/class_map.csv"), crop) if crop else None
    if keep is not None:
        print(f"{crop}: {len(keep)} PlantVillage classes", flush=True)

    images = [
        p
        for p in sorted(image_root.rglob("*"))
        if p.suffix in IMAGE_EXTS and p.is_file() and (keep is None or p.parent.name in keep)
    ]
    if args.limit:
        images = images[: args.limit]
    print(f"{len(images)} images to pseudo-label", flush=True)
    if not images:
        raise ValueError(f"No images matched under {image_root} for crop={crop!r}")

    model, size = load_segmenter(ckpt, device)

    csv_path = out_root / "pseudo_confidence.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    written, no_leaf, empty = 0, 0, 0
    t0 = time.time()

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["path", "class_name", "mean_fg_prob", "fg_fraction", "leaf_coverage", "clipped_away"]
        )

        for i, src in enumerate(images):
            dataset, relative = split_raw_path(src)
            out = out_root / dataset / f"{relative.as_posix()}.png"
            if out.exists() and out.stat().st_size > 0:
                continue

            bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if bgr is None:
                raise OSError(f"Could not read image: {src}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            leaf_path = mask_path_for(src, leaf_root)
            if not leaf_path.exists():
                no_leaf += 1
                continue
            leaf = cv2.imread(str(leaf_path), cv2.IMREAD_GRAYSCALE)
            if leaf is None:
                raise OSError(f"Leaf mask exists but could not be read: {leaf_path}")
            leaf = (leaf > 127).astype(np.uint8)
            if leaf.shape[:2] != rgb.shape[:2]:
                leaf = cv2.resize(
                    leaf, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
                )

            probs = predict_probs(model, rgb, size, device)
            raw = (probs > threshold).astype(np.uint8)
            clipped = (raw & leaf).astype(np.uint8)

            raw_n = int(raw.sum())
            removed = (raw_n - int(clipped.sum())) / max(raw_n, 1)

            fg_fraction = float(clipped.mean())
            mean_fg_prob = float(probs[clipped > 0].mean()) if clipped.any() else 0.0
            if fg_fraction == 0.0:
                empty += 1

            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), clipped * 255)
            written += 1

            w.writerow(
                [
                    relative.as_posix(),
                    src.parent.name,
                    round(mean_fg_prob, 4),
                    round(fg_fraction, 5),
                    round(float(leaf.mean()), 4),
                    round(removed, 4),
                ]
            )

            if i % 200 == 0:
                rate = (i + 1) / max(time.time() - t0, 1e-6)
                print(
                    f"  {i + 1}/{len(images)} {rate:.1f} img/s "
                    f"eta {(len(images) - i - 1) / max(rate, 1e-6) / 60:.1f} min",
                    flush=True,
                )
                fh.flush()

    print(
        f"\ndone: {written} pseudo masks in {(time.time() - t0) / 60:.1f} min\n"
        f"  empty (no lesion predicted): {empty}\n"
        f"  skipped (leaf mask missing): {no_leaf}\n"
        f"  confidence csv: {csv_path}",
        flush=True,
    )
    if no_leaf:
        print(
            f"WARNING: {no_leaf} images had no leaf mask. Run masks/leaf_masks.py first "
            "or those images will be absent from the pseudo set.",
            flush=True,
        )


if __name__ == "__main__":
    main()
