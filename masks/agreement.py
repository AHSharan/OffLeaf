"""Inter-annotator agreement between two sets of human lesion masks.

Answers "do two people drawing the same lesion produce the same mask?". That
number is the ceiling on what any segmenter can be expected to achieve, and the
context in which the S0 val IoU should be read: a model scoring 0.55 against
annotators who agree with each other at 0.60 is close to the practical limit,
whereas the same 0.55 against annotators agreeing at 0.90 means real headroom.

Compares only images present in **both** directories. Coverage of the overlap
is reported so a tiny, unrepresentative intersection is visible rather than
being quietly averaged into a confident-looking number.

Usage::

    python masks/agreement.py --a data/masks/annotator_a --b data/masks/annotator_b
    python masks/agreement.py --a ... --b ... --out masks/agreement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _load(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Could not read mask: {path}")
    if shape_hw is not None and m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def iou_and_dice(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """IoU and Dice for two binary masks. Both empty counts as perfect agreement."""
    ab = np.logical_and(a > 0, b > 0).sum()
    union = np.logical_or(a > 0, b > 0).sum()
    total = (a > 0).sum() + (b > 0).sum()
    iou = 1.0 if union == 0 else float(ab / union)
    dice = 1.0 if total == 0 else float(2 * ab / total)
    return iou, dice


def compare(dir_a: Path, dir_b: Path) -> dict:
    """Compare every mask present in both directories.

    Raises:
        FileNotFoundError: If either directory is missing.
        ValueError: If they share no masks.
    """
    for d in (dir_a, dir_b):
        if not d.exists():
            raise FileNotFoundError(f"Annotator directory not found: {d}")

    rel_a = {p.relative_to(dir_a).as_posix() for p in dir_a.rglob("*.png")}
    rel_b = {p.relative_to(dir_b).as_posix() for p in dir_b.rglob("*.png")}
    shared = sorted(rel_a & rel_b)
    if not shared:
        raise ValueError(
            f"No overlapping masks between {dir_a} and {dir_b}. "
            f"({len(rel_a)} and {len(rel_b)} masks respectively, none in common.)"
        )

    ious, dices, per_image = [], [], []
    for rel in shared:
        a = _load(dir_a / rel)
        b = _load(dir_b / rel, a.shape[:2])
        iou, dice = iou_and_dice(a, b)
        ious.append(iou)
        dices.append(dice)
        per_image.append(
            {
                "id": rel,
                "iou": round(iou, 4),
                "dice": round(dice, 4),
                "coverage_a": round(float(a.mean()), 4),
                "coverage_b": round(float(b.mean()), 4),
            }
        )

    arr = np.array(ious)
    worst = sorted(per_image, key=lambda r: r["iou"])[:10]

    return {
        "annotator_a": str(dir_a),
        "annotator_b": str(dir_b),
        "masks_a_total": len(rel_a),
        "masks_b_total": len(rel_b),
        "overlapping_masks": len(shared),
        "overlap_fraction_of_a": round(len(shared) / max(len(rel_a), 1), 4),
        "mean_iou": round(float(arr.mean()), 4),
        "median_iou": round(float(np.median(arr)), 4),
        "std_iou": round(float(arr.std()), 4),
        "min_iou": round(float(arr.min()), 4),
        "mean_dice": round(float(np.mean(dices)), 4),
        "frac_below_0.5_iou": round(float((arr < 0.5).mean()), 4),
        "worst_10": worst,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Inter-annotator agreement on lesion masks.")
    ap.add_argument("--a", required=True, help="first annotator's mask directory")
    ap.add_argument("--b", required=True, help="second annotator's mask directory")
    ap.add_argument("--out", default=None, help="where to write the JSON summary")
    args = ap.parse_args()

    summary = compare(Path(args.a), Path(args.b))
    printable = {k: v for k, v in summary.items() if k != "worst_10"}
    print(json.dumps(printable, indent=2))

    print("\nworst 10 by IoU:")
    for r in summary["worst_10"]:
        print(f"  {r['iou']:.3f}  cov {r['coverage_a']:.3f} vs {r['coverage_b']:.3f}  {r['id']}")

    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "agreement.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if summary["overlapping_masks"] < 20:
        print(
            f"\nWARNING: only {summary['overlapping_masks']} overlapping mask(s). "
            "Agreement estimated on this few images is not a stable number - have both "
            "annotators cover a larger shared subset before quoting it.",
            flush=True,
        )


if __name__ == "__main__":
    main()
