"""Promote verified pseudo masks into ``lesion_human/``.

Reads the ``verify.csv`` a human filled in after reviewing
``verify_sheet.html`` and acts on each decision:

* ``accept`` - the pseudo mask was judged correct; copy it into
  ``lesion_human/``. It is now human-*verified* and metric-grade.
* ``fix``    - the annotator redrew it (CVAT export); copy the corrected mask
  in, and record IoU against the original pseudo mask.
* ``reject`` - the pseudo mask was wrong; nothing is promoted.

The pseudo-vs-human IoU on the fixed subset is the honest read on segmenter
quality: accepted masks are biased upward (they were accepted *because* they
looked right), so IoU over the fixed ones is what tells you how wrong the model
is when it is wrong. Written to ``masks/quality.json``.

Usage::

    python masks/import_verified.py --config configs/verify_tomato.yaml
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

VALID_DECISIONS = {"accept", "fix", "reject"}


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two binary masks. Returns 1.0 when both are empty."""
    a = a > 0
    b = b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def _read_binary(path: Path, shape_hw: tuple[int, int] | None = None) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Could not read mask: {path}")
    if shape_hw is not None and m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def promote(
    verify_dir: Path,
    pseudo_root: Path,
    human_root: Path,
    fixed_dir: Path,
    dataset: str = "plantvillage",
) -> dict:
    """Act on the decisions in ``verify_dir/verify.csv``; return the summary.

    Separated from :func:`main` so ``tests/test_pseudo_import_iou.py`` can drive
    it against a temporary tree without constructing configs or touching the
    real dataset.

    Raises:
        FileNotFoundError: If ``verify.csv`` or an accepted pseudo mask is absent.
        ValueError: If the decisions are unfilled or contain unknown values.
    """
    vcsv = verify_dir / "verify.csv"
    if not vcsv.exists():
        raise FileNotFoundError(
            f"{vcsv} not found. Run masks/verify_sheet.py and fill in the decisions first."
        )

    df = pd.read_csv(vcsv).fillna({"decision": "", "note": ""})
    df["decision"] = df.decision.astype(str).str.strip().str.lower()

    blank = int((df.decision == "").sum())
    bad = sorted(set(df.decision) - VALID_DECISIONS - {""})
    if bad:
        raise ValueError(
            f"Unrecognised decision value(s) in {vcsv}: {bad}. "
            f"Use one of {sorted(VALID_DECISIONS)}."
        )
    if blank == len(df):
        raise ValueError(
            f"No decisions filled in {vcsv}. Open {verify_dir / 'verify_sheet.html'}, "
            "review each mask, then fill the 'decision' column."
        )

    counts: Counter = Counter(df.decision)
    promoted, fixed_ious, missing_fixed = 0, [], []

    for row in df.to_dict("records"):
        decision, rel = row["decision"], str(row["id"])
        if decision not in {"accept", "fix"}:
            continue

        pseudo_path = pseudo_root / dataset / f"{rel}.png"
        out_path = human_root / dataset / f"{rel}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if decision == "accept":
            if not pseudo_path.exists():
                raise FileNotFoundError(f"Accepted mask missing from pseudo set: {pseudo_path}")
            mask = _read_binary(pseudo_path)
            cv2.imwrite(str(out_path), mask * 255)
            promoted += 1
            continue

        # decision == "fix": the corrected mask comes from outside the pseudo set.
        candidate = fixed_dir / f"{rel}.png"
        if not candidate.exists():
            candidate = fixed_dir / f"{Path(rel).name}.png"
        if not candidate.exists():
            missing_fixed.append(rel)
            continue

        corrected = _read_binary(candidate)
        cv2.imwrite(str(out_path), corrected * 255)
        promoted += 1

        if pseudo_path.exists():
            pseudo = _read_binary(pseudo_path, corrected.shape[:2])
            fixed_ious.append(binary_iou(pseudo, corrected))

    summary = {
        "verify_csv": vcsv.as_posix(),
        "rows": int(len(df)),
        "decisions": dict(counts),
        "blank_decisions": blank,
        "promoted_to_lesion_human": promoted,
        "fixed_with_corrected_mask": len(fixed_ious),
        "missing_corrected_masks": len(missing_fixed),
        "missing_corrected_examples": missing_fixed[:5],
        # The headline number: how close the segmenter was on the cases a human
        # judged wrong enough to redraw.
        "pseudo_vs_human_iou_on_fixed": round(float(np.mean(fixed_ious)), 4) if fixed_ious else None,
        "pseudo_vs_human_iou_min": round(float(np.min(fixed_ious)), 4) if fixed_ious else None,
        "accept_rate": round(counts.get("accept", 0) / max(len(df) - blank, 1), 4),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Promote verified masks into lesion_human/.")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--fixed_dir",
        default=None,
        help="directory of corrected masks for 'fix' rows (e.g. a CVAT export)",
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)

    dataset = cfg.get("dataset_name", "plantvillage")
    pseudo_root = repo / cfg.get("pseudo_root", "data/masks/lesion_pseudo")
    human_root = repo / cfg.get("human_root", "data/masks/lesion_human")
    verify_dir = repo / cfg.get("out_dir", "data/masks/verify")
    fixed_dir = (
        Path(args.fixed_dir)
        if args.fixed_dir
        else repo / cfg.get("fixed_dir", "data/masks/verify/fixed")
    )

    summary = promote(verify_dir, pseudo_root, human_root, fixed_dir, dataset)

    print(json.dumps(summary, indent=2))

    qpath = repo / "masks" / "quality.json"
    existing = json.loads(qpath.read_text(encoding="utf-8")) if qpath.exists() else {}
    existing["import_verified"] = summary
    qpath.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nwrote {qpath}")

    if summary["missing_corrected_masks"]:
        print(
            f"\nWARNING: {summary['missing_corrected_masks']} row(s) marked 'fix' had no "
            f"corrected mask in "
            f"{fixed_dir}. Export them from CVAT (masks/cvat_import.py) and re-run.",
            flush=True,
        )


if __name__ == "__main__":
    main()
