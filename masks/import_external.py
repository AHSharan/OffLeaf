"""Import externally-annotated lesion masks into ``data/masks/lesion_human/``.

These masks were drawn by people, so they land in ``lesion_human/`` and are
valid for reported metrics. Model-generated masks must never be written here —
they go to ``lesion_pseudo/``. See CLAUDE.md section 6.

**PlantSeg encoding** (verified, not assumed): each annotation PNG is
single-channel uint8 at the image's own resolution, containing exactly one
non-zero value which is that disease's class index (e.g. 97 for tomato
bacterial leaf spot, 1 for apple black rot). Indices are consistent per disease
and no sampled file contained two classes, so binarising on ``> 0`` is correct.

Masks mirror the raw tree, matching ``data/dataset.py``::

    data/raw/plantseg/plantsegv2/images/train/tomato_late_blight_1.jpg
    data/masks/lesion_human/plantseg/plantsegv2/images/train/tomato_late_blight_1.jpg.png

Usage::

    python masks/import_external.py --config configs/import_plantseg.yaml
    python masks/import_external.py --config configs/import_plantseg.yaml --crop tomato
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config  # noqa: E402

LESION_HUMAN_ROOT = "data/masks/lesion_human"


def import_plantseg(
    plantseg_root: Path,
    out_root: Path,
    crop: str | None = None,
    overwrite: bool = False,
) -> dict:
    """Convert PlantSeg annotations to binary lesion masks.

    Args:
        plantseg_root: ``data/raw/plantseg`` (containing ``plantsegv2/``).
        out_root: ``data/masks/lesion_human``.
        crop: Optional lowercase plant filter, e.g. ``"tomato"``.
        overwrite: Re-write masks that already exist.

    Returns:
        Summary dict, also written to ``quality.json`` by the caller.

    Raises:
        FileNotFoundError: If the metadata or the expected subfolders are absent.
    """
    v2 = plantseg_root / "plantsegv2"
    meta_path = v2 / "Metadatav2.csv"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"PlantSeg metadata not found at {meta_path}. Download with:\n"
            "  kaggle datasets download weitianqi/plantseg -p data/raw/plantseg --unzip"
        )

    df = pd.read_csv(meta_path)
    if crop:
        df = df[df.Plant.str.lower() == crop.lower()]
        if df.empty:
            raise ValueError(f"No PlantSeg rows for crop {crop!r}")

    split_dir = {"Training": "train", "Validation": "val", "Test": "test"}

    written, skipped, empty = 0, 0, 0
    missing_image: list[str] = []
    missing_ann: list[str] = []
    per_disease: Counter = Counter()
    ratios: list[float] = []

    required = {"Name", "Label file", "Split", "Plant", "Disease"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"PlantSeg metadata is missing expected columns: {sorted(missing_cols)}")

    # dict records, not itertuples: the "Label file" column contains a space,
    # which itertuples silently renames to a positional attribute.
    for row in df.to_dict("records"):
        sub = split_dir.get(row["Split"])
        if sub is None:
            raise ValueError(f"Unexpected Split value {row['Split']!r} in PlantSeg metadata")

        img_path = v2 / "images" / sub / str(row["Name"])
        ann_path = v2 / "annotations" / sub / str(row["Label file"])

        if not img_path.exists():
            missing_image.append(str(img_path.relative_to(plantseg_root)))
            continue
        if not ann_path.exists():
            missing_ann.append(str(ann_path.relative_to(plantseg_root)))
            continue

        rel = img_path.relative_to(plantseg_root)
        out = out_root / "plantseg" / f"{rel.as_posix()}.png"
        if out.exists() and out.stat().st_size > 0 and not overwrite:
            skipped += 1
            continue

        ann = cv2.imread(str(ann_path), cv2.IMREAD_UNCHANGED)
        if ann is None:
            raise OSError(f"Annotation exists but could not be read: {ann_path}")
        if ann.ndim == 3:
            ann = ann[..., 0]

        binary = (ann > 0).astype(np.uint8)

        # Resolution must match the image, or every downstream mass metric is
        # computed against a shifted target.
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise OSError(f"Image exists but could not be read: {img_path}")
        if binary.shape[:2] != img.shape[:2]:
            binary = cv2.resize(
                binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
            )

        ratio = float(binary.mean())
        if ratio == 0.0:
            empty += 1
        ratios.append(ratio)
        per_disease[str(row["Disease"])] += 1

        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), binary * 255)
        written += 1

    return {
        "source": "plantseg",
        "crop_filter": crop,
        "rows_considered": int(len(df)),
        "masks_written": written,
        "already_present_skipped": skipped,
        "empty_masks": empty,
        "missing_images": len(missing_image),
        "missing_annotations": len(missing_ann),
        "missing_image_examples": missing_image[:5],
        "missing_annotation_examples": missing_ann[:5],
        "mean_lesion_ratio": round(float(np.mean(ratios)), 4) if ratios else None,
        "per_disease": dict(per_disease.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Import external lesion masks into lesion_human/.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--crop", default=None, help="only this plant, e.g. tomato")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)

    plantseg_root = repo / cfg.get("plantseg_root", "data/raw/plantseg")
    out_root = repo / cfg.get("out_root", LESION_HUMAN_ROOT)
    crop = args.crop if args.crop is not None else cfg.get("crop")

    summary = import_plantseg(plantseg_root, out_root, crop=crop, overwrite=args.overwrite)

    print(json.dumps(summary, indent=2))

    qpath = repo / "masks" / "quality.json"
    existing = {}
    if qpath.exists():
        existing = json.loads(qpath.read_text(encoding="utf-8"))
    existing["import_external"] = summary
    qpath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote {qpath}")


if __name__ == "__main__":
    main()
