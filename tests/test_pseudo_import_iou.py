"""Spec section 10: import_verified must report IoU on the fixed masks.

The fixed-subset IoU is the only unbiased read on segmenter quality available
from a verification pass. Accepted masks were accepted *because* they looked
right, so their agreement is inflated by construction; the redrawn ones measure
how wrong the model is when it is wrong. If that number silently went missing,
the pseudo-labelling pipeline would have no honest quality signal at all.
"""

from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from masks.import_verified import binary_iou, promote


def _write_mask(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), arr.astype(np.uint8) * 255)


@pytest.fixture
def verify_tree(tmp_path):
    """A miniature verification tree with known-IoU fix cases."""
    dataset = "plantvillage"
    pseudo_root = tmp_path / "lesion_pseudo"
    human_root = tmp_path / "lesion_human"
    fixed_dir = tmp_path / "fixed"
    verify_dir = tmp_path / "verify"
    verify_dir.mkdir(parents=True, exist_ok=True)

    size = 40
    rows = []

    # accept x2 - pseudo mask copied through unchanged
    for i in range(2):
        rel = f"color/cls_a/accept_{i}.jpg"
        m = np.zeros((size, size), np.uint8)
        m[10:30, 10:30] = 1
        _write_mask(pseudo_root / dataset / f"{rel}.png", m)
        rows.append({"id": rel, "class_name": "cls_a", "decision": "accept", "note": ""})

    # fix x2 - corrected mask differs from pseudo by a known amount.
    # pseudo = rows 10:30, corrected = rows 10:20 (half the area, fully contained)
    # => intersection 10*20, union 20*20  => IoU exactly 0.5
    for i in range(2):
        rel = f"color/cls_b/fix_{i}.jpg"
        pseudo = np.zeros((size, size), np.uint8)
        pseudo[10:30, 10:30] = 1
        corrected = np.zeros((size, size), np.uint8)
        corrected[10:20, 10:30] = 1
        _write_mask(pseudo_root / dataset / f"{rel}.png", pseudo)
        _write_mask(fixed_dir / f"{rel}.png", corrected)
        rows.append({"id": rel, "class_name": "cls_b", "decision": "fix", "note": "redrawn"})

    # reject x1 - nothing should be promoted
    rel = "color/cls_c/reject_0.jpg"
    m = np.zeros((size, size), np.uint8)
    m[0:5, 0:5] = 1
    _write_mask(pseudo_root / dataset / f"{rel}.png", m)
    rows.append({"id": rel, "class_name": "cls_c", "decision": "reject", "note": "wrong"})

    with open(verify_dir / "verify.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "class_name", "decision", "note"])
        w.writeheader()
        w.writerows(rows)

    return {
        "verify_dir": verify_dir,
        "pseudo_root": pseudo_root,
        "human_root": human_root,
        "fixed_dir": fixed_dir,
        "dataset": dataset,
    }


def test_binary_iou_known_values():
    a = np.zeros((10, 10), np.uint8)
    b = np.zeros((10, 10), np.uint8)
    a[0:10, 0:10] = 1
    b[0:5, 0:10] = 1
    assert binary_iou(a, b) == pytest.approx(0.5)
    assert binary_iou(a, a) == pytest.approx(1.0)
    assert binary_iou(np.zeros((4, 4)), np.zeros((4, 4))) == pytest.approx(1.0)


def test_pseudo_import_iou(verify_tree):
    """IoU on the fixed subset is computed and is the value we constructed."""
    s = promote(
        verify_tree["verify_dir"],
        verify_tree["pseudo_root"],
        verify_tree["human_root"],
        verify_tree["fixed_dir"],
        verify_tree["dataset"],
    )

    assert s["pseudo_vs_human_iou_on_fixed"] is not None, "fixed-subset IoU was not reported"
    assert s["pseudo_vs_human_iou_on_fixed"] == pytest.approx(0.5, abs=1e-3), (
        f"expected IoU 0.5 by construction, got {s['pseudo_vs_human_iou_on_fixed']}"
    )
    assert s["fixed_with_corrected_mask"] == 2
    assert s["promoted_to_lesion_human"] == 4  # 2 accepted + 2 fixed, never the rejected one
    assert s["decisions"]["reject"] == 1


def test_rejected_mask_is_not_promoted(verify_tree):
    """A rejected mask must never reach lesion_human/."""
    promote(
        verify_tree["verify_dir"],
        verify_tree["pseudo_root"],
        verify_tree["human_root"],
        verify_tree["fixed_dir"],
        verify_tree["dataset"],
    )
    rejected = (
        verify_tree["human_root"]
        / verify_tree["dataset"]
        / "color/cls_c/reject_0.jpg.png"
    )
    assert not rejected.exists(), "a rejected pseudo mask was promoted to lesion_human/"


def test_quality_json_is_serialisable(verify_tree):
    """The summary must round-trip through JSON - it is written to quality.json."""
    s = promote(
        verify_tree["verify_dir"],
        verify_tree["pseudo_root"],
        verify_tree["human_root"],
        verify_tree["fixed_dir"],
        verify_tree["dataset"],
    )
    reloaded = json.loads(json.dumps(s))
    assert reloaded["pseudo_vs_human_iou_on_fixed"] == pytest.approx(0.5, abs=1e-3)


def test_unfilled_decisions_raise(tmp_path):
    """An unreviewed verify.csv must fail loudly, not promote nothing quietly."""
    vd = tmp_path / "verify"
    vd.mkdir()
    with open(vd / "verify.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "class_name", "decision", "note"])
        w.writeheader()
        w.writerow({"id": "a.jpg", "class_name": "c", "decision": "", "note": ""})

    with pytest.raises(ValueError, match="No decisions filled"):
        promote(vd, tmp_path / "p", tmp_path / "h", tmp_path / "f")


def test_unknown_decision_raises(tmp_path):
    """A typo'd decision must not be silently treated as reject."""
    vd = tmp_path / "verify"
    vd.mkdir()
    with open(vd / "verify.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "class_name", "decision", "note"])
        w.writeheader()
        w.writerow({"id": "a.jpg", "class_name": "c", "decision": "acept", "note": ""})

    with pytest.raises(ValueError, match="Unrecognised decision"):
        promote(vd, tmp_path / "p", tmp_path / "h", tmp_path / "f")
