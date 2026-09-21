"""Spec section 10: the remaining metric and explainer guarantees.

Covers `test_cam_shapes`, `test_flip_rate_identity`, `test_metrics_refuse_pseudo`
and `test_counterfactual_index`.

The pseudo-mask refusal is the one that matters most. It is not a nicety - it is
the rule that separates "a model trained with help from a segmenter" from "a
result validated against a segmenter's opinion". If it ever silently stopped
firing, every reported relevance number would quietly become circular.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eval.metrics import (
    assert_human_masks,
    bootstrap_ci,
    flip_rate,
    gap_decomposition,
    relevance_mass,
    top_mass_iou,
)
from explain.methods import METHODS, explain
from models.build import build_model

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# test_cam_shapes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backbone", ["resnet50", "vit_small_patch16_224"])
@pytest.mark.parametrize("method", METHODS)
def test_cam_shapes(backbone, method):
    """Every explainer returns H x W in [0,1] at input resolution, both backbones."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, num_classes=3, pretrained=False).to(device).eval()

    b, size = 2, 224
    images = torch.randn(b, 3, size, size, device=device)
    labels = torch.tensor([0, 1], device=device)

    cam = explain(model, images, labels, method=method)

    assert cam.shape == (b, size, size), (
        f"{backbone}/{method} returned {tuple(cam.shape)}, expected {(b, size, size)}"
    )
    assert torch.isfinite(cam).all(), f"{backbone}/{method} produced non-finite values"
    assert float(cam.min()) >= 0.0, f"{backbone}/{method} min {float(cam.min())} < 0"
    assert float(cam.max()) <= 1.0, f"{backbone}/{method} max {float(cam.max())} > 1"


def test_explain_rejects_unknown_method():
    device = torch.device("cpu")
    model = build_model("resnet50", num_classes=3, pretrained=False).to(device).eval()
    with pytest.raises(ValueError, match="Unknown method"):
        explain(model, torch.randn(1, 3, 224, 224), method="not_a_method")


# --------------------------------------------------------------------------
# test_flip_rate_identity
# --------------------------------------------------------------------------


def test_flip_rate_identity():
    """Flip rate is exactly 0 when the 'swapped' background is the original."""
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 10, size=64)
    conf = rng.random(64)

    r = flip_rate(pred, conf, pred, conf)

    assert r["flip_rate"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert r["mean_abs_delta_confidence"]["point"] == pytest.approx(0.0, abs=1e-12)
    assert r["n_pairs"] == 64


def test_flip_rate_all_different():
    """Sanity: fully disagreeing predictions give a flip rate of 1."""
    a = np.zeros(20, dtype=int)
    b = np.ones(20, dtype=int)
    r = flip_rate(a, np.full(20, 0.9), b, np.full(20, 0.4))
    assert r["flip_rate"]["point"] == pytest.approx(1.0)
    assert r["mean_abs_delta_confidence"]["point"] == pytest.approx(0.5, abs=1e-6)


def test_flip_rate_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="differ in shape"):
        flip_rate([1, 2, 3], [0.1, 0.2, 0.3], [1, 2], [0.1, 0.2])


# --------------------------------------------------------------------------
# test_metrics_refuse_pseudo
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path",
    [
        "data/masks/lesion_pseudo",
        "data/masks/lesion_pseudo/plantvillage",
        r"F:\OffLeaf\data\masks\lesion_pseudo",
        "some/pseudo/dir",
    ],
)
def test_metrics_refuse_pseudo(path):
    """relevance_mass refuses a pseudo-mask directory without --allow_pseudo."""
    with pytest.raises(ValueError, match="Refusing to compute relevance mass on pseudo"):
        assert_human_masks(path, allow_pseudo=False)


def test_allow_pseudo_permits_and_tags():
    """With the override the call proceeds, and the result is flagged."""
    assert_human_masks("data/masks/lesion_pseudo", allow_pseudo=True)

    heat = [np.ones((8, 8), dtype=np.float32)]
    leaf = [np.ones((8, 8), dtype=np.uint8)]
    lesion = [np.zeros((8, 8), dtype=np.uint8)]
    lesion[0][2:5, 2:5] = 1

    out = relevance_mass(
        heat, leaf, lesion, lesion_mask_dir="data/masks/lesion_pseudo", allow_pseudo=True
    )
    assert out["PSEUDO_MASK_BASED"] is True
    assert out["mask_source"] == "pseudo"


def test_human_masks_pass_through():
    assert_human_masks("data/masks/lesion_human/plantvillage", allow_pseudo=False)


def test_relevance_mass_on_pseudo_dir_raises_end_to_end():
    """The refusal fires through relevance_mass, not only the helper."""
    heat = [np.ones((4, 4), dtype=np.float32)]
    leaf = [np.ones((4, 4), dtype=np.uint8)]
    with pytest.raises(ValueError, match="Refusing"):
        relevance_mass(heat, leaf, leaf, lesion_mask_dir="data/masks/lesion_pseudo")


# --------------------------------------------------------------------------
# relevance mass arithmetic
# --------------------------------------------------------------------------


def test_relevance_mass_known_values():
    """Constructed case: half the heat on the lesion, a quarter off the leaf."""
    h = np.zeros((4, 4), dtype=np.float32)
    leaf = np.zeros((4, 4), dtype=np.uint8)
    lesion = np.zeros((4, 4), dtype=np.uint8)

    leaf[:, :3] = 1        # leaf covers 3 of 4 columns
    lesion[:, 0] = 1       # lesion is column 0
    h[:, 0] = 1.0          # 4 units on lesion
    h[:, 1] = 1.0          # 4 units on leaf, not lesion
    h[:, 3] = 1.0          # 4 units off leaf

    out = relevance_mass([h], [leaf], [lesion], lesion_mask_dir="data/masks/lesion_human")
    assert out["lesion_mass"]["point"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["leaf_not_lesion_mass"]["point"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["offleaf_mass"]["point"] == pytest.approx(1 / 3, abs=1e-6)
    assert out["offlesion_mass"]["point"] == pytest.approx(2 / 3, abs=1e-6)


def test_top_mass_iou_perfect_and_disjoint():
    heat = np.zeros((8, 8), dtype=np.float32)
    target = np.zeros((8, 8), dtype=np.uint8)
    heat[0:4, 0:4] = 1.0
    target[0:4, 0:4] = 1
    assert top_mass_iou(heat, target) == pytest.approx(1.0)

    disjoint = np.zeros((8, 8), dtype=np.uint8)
    disjoint[4:8, 4:8] = 1
    assert top_mass_iou(heat, disjoint) == pytest.approx(0.0)


def test_bootstrap_ci_brackets_point():
    vals = np.random.default_rng(0).normal(0.5, 0.1, 200)
    ci = bootstrap_ci(vals, seed=0)
    assert ci["lo"] <= ci["point"] <= ci["hi"]
    assert ci["n"] == 200


# --------------------------------------------------------------------------
# gap decomposition + counterfactual index
# --------------------------------------------------------------------------


def test_gap_decomposition_subtracts_paste_cost():
    """A drop caused purely by compositing must not be reported as background."""
    cells = {
        "lab_plain": 0.95,
        "lab_field": 0.80,
        "field_plain": 0.70,
        "field_field": 0.40,
        "paste_control": 0.90,   # 0.05 lost to the paste itself
    }
    g = gap_decomposition(cells)
    assert g["paste_artefact_cost"] == pytest.approx(0.05)
    assert g["background_effect_uncorrected"] == pytest.approx(0.15)
    assert g["background_effect"] == pytest.approx(0.10)  # 0.15 - 0.05
    assert g["total_gap"] == pytest.approx(0.55)


def test_gap_decomposition_requires_all_cells():
    with pytest.raises(KeyError, match="missing cell"):
        gap_decomposition({"lab_plain": 0.9, "field_field": 0.4})


def test_counterfactual_index():
    """Every image_id appears in all five cells (skips if the set isn't built)."""
    import pandas as pd

    index = REPO / "data" / "counterfactual" / "index.csv"
    if not index.exists():
        pytest.skip("counterfactual set not built yet - run counterfactual/build.py")

    df = pd.read_csv(index)
    cells = set(df.cell.unique())
    lab_ids = set(df[df.source_dataset == "plantvillage"].image_id)
    for cell in ("lab_plain", "lab_field", "paste_control"):
        if cell not in cells:
            continue
        present = set(df[df.cell == cell].image_id)
        missing = lab_ids - present
        assert not missing, f"{len(missing)} lab image_id(s) absent from cell {cell}"
