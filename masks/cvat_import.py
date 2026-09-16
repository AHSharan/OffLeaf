"""Import a CVAT export into ``data/masks/lesion_human/``.

Two export formats are handled, because CVAT offers both and annotators pick
whichever the task was set up with:

* **COCO 1.0** - ``instances_default.json`` with polygon or RLE segmentations.
  Multiple annotations for one image are unioned into a single binary mask.
* **Segmentation mask 1.1** - a directory of indexed PNGs. Any non-zero pixel
  becomes lesion, matching the PlantSeg convention.

Masks written here are human-drawn, so they are metric-grade. Filenames must
resolve to a real image under the dataset root - an annotation whose image
cannot be located is reported, never silently dropped, because a silent drop
would quietly shrink the evaluation set.

Usage::

    python masks/cvat_import.py --config configs/cvat_import.yaml
    python masks/cvat_import.py --config configs/cvat_import.yaml --format masks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


def build_image_index(image_root: Path) -> dict[str, Path]:
    """Map bare filename -> path, for resolving CVAT's flattened names.

    CVAT exports usually keep only the basename, so the original folder
    structure has to be recovered by lookup. Collisions are reported by the
    caller rather than silently resolved.
    """
    index: dict[str, Path] = {}
    collisions: set[str] = set()
    for p in image_root.rglob("*"):
        if p.suffix in IMAGE_EXTS and p.is_file():
            if p.name in index:
                collisions.add(p.name)
            index[p.name] = p
    if collisions:
        print(
            f"WARNING: {len(collisions)} filename(s) occur in more than one folder under "
            f"{image_root}; the last match wins. Examples: {sorted(collisions)[:3]}",
            flush=True,
        )
    return index


def _poly_to_mask(seg, h: int, w: int) -> np.ndarray:
    """Rasterise one COCO segmentation (polygon list or RLE) to a binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if isinstance(seg, list):
        for poly in seg:
            if len(poly) < 6:  # need at least 3 points
                continue
            pts = np.array(poly, dtype=np.float64).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)
        return mask

    if isinstance(seg, dict) and "counts" in seg:
        try:
            from pycocotools import mask as coco_mask  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "This CVAT export uses RLE segmentations, which need pycocotools.\n"
                "  pip install pycocotools"
            ) from exc
        return coco_mask.decode(seg).astype(np.uint8)

    raise ValueError(f"Unrecognised COCO segmentation format: {type(seg)}")


def import_coco(json_path: Path, image_root: Path, out_root: Path, dataset: str) -> dict:
    """Import a COCO-format CVAT export."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    images = {im["id"]: im for im in data.get("images", [])}
    if not images:
        raise ValueError(f"No images listed in {json_path}")

    by_image: dict[int, list] = {}
    for ann in data.get("annotations", []):
        if ann.get("segmentation"):
            by_image.setdefault(ann["image_id"], []).append(ann)

    index = build_image_index(image_root)
    written, unresolved, empty = 0, [], 0

    for img_id, meta in images.items():
        anns = by_image.get(img_id, [])
        name = Path(meta["file_name"]).name
        src = index.get(name)
        if src is None:
            unresolved.append(meta["file_name"])
            continue

        h = int(meta.get("height") or 0)
        w = int(meta.get("width") or 0)
        if not (h and w):
            probe = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if probe is None:
                raise OSError(f"Could not read image to determine size: {src}")
            h, w = probe.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            mask |= _poly_to_mask(ann["segmentation"], h, w)

        if not mask.any():
            empty += 1

        rel = src.relative_to(image_root)
        out = out_root / dataset / f"{rel.as_posix()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mask * 255)
        written += 1

    return {
        "format": "coco",
        "source": str(json_path),
        "images_in_export": len(images),
        "masks_written": written,
        "empty_masks": empty,
        "unresolved_images": len(unresolved),
        "unresolved_examples": unresolved[:5],
    }


def import_mask_pngs(mask_dir: Path, image_root: Path, out_root: Path, dataset: str) -> dict:
    """Import a CVAT 'segmentation mask' export (a directory of indexed PNGs)."""
    index = build_image_index(image_root)
    pngs = sorted(p for p in mask_dir.rglob("*.png"))
    if not pngs:
        raise ValueError(f"No PNG masks found under {mask_dir}")

    written, unresolved, empty = 0, [], 0
    for p in pngs:
        stem = p.stem
        src = index.get(stem) or next(
            (index[k] for k in index if Path(k).stem == stem), None
        )
        if src is None:
            unresolved.append(p.name)
            continue

        m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if m is None:
            raise OSError(f"Could not read CVAT mask: {p}")
        if m.ndim == 3:
            m = m.max(axis=2)
        mask = (m > 0).astype(np.uint8)

        img = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Could not read image: {src}")
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(
                mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        if not mask.any():
            empty += 1

        rel = src.relative_to(image_root)
        out = out_root / dataset / f"{rel.as_posix()}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), mask * 255)
        written += 1

    return {
        "format": "mask_png",
        "source": str(mask_dir),
        "masks_in_export": len(pngs),
        "masks_written": written,
        "empty_masks": empty,
        "unresolved_images": len(unresolved),
        "unresolved_examples": unresolved[:5],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Import a CVAT export into lesion_human/.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--format", choices=["coco", "masks"], default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)

    image_root = repo / cfg["image_root"]
    out_root = repo / cfg.get("out_root", "data/masks/lesion_human")
    dataset = cfg.get("dataset_name", "plantvillage")
    fmt = args.format or cfg.get("format", "coco")

    if fmt == "coco":
        summary = import_coco(repo / cfg["coco_json"], image_root, out_root, dataset)
    else:
        summary = import_mask_pngs(repo / cfg["mask_dir"], image_root, out_root, dataset)

    print(json.dumps(summary, indent=2))

    qpath = repo / "masks" / "quality.json"
    existing = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else {}
    existing["cvat_import"] = summary
    qpath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote {qpath}")

    if summary["unresolved_images"]:
        print(
            f"\nWARNING: {summary['unresolved_images']} annotation(s) could not be matched to an "
            f"image under {image_root}. They were NOT imported. Check the export's file names.",
            flush=True,
        )


if __name__ == "__main__":
    main()
