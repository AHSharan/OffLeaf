"""Spec section 10: the training transform must move image and mask identically.

If a random crop moved the image but not the mask, the CAM penalty would be
supervised against a misaligned target and every E1d/E2/E3 result would be
quietly wrong while still training and reporting plausible accuracy. This is the
single most load-bearing invariant in the data pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, get_transforms
from tests.conftest import binary_iou


def _recover_square(image_tensor) -> np.ndarray:
    """Recover the white square from a normalised CHW tensor.

    Undoes the ImageNet normalisation and thresholds at mid-grey. The square is
    pure white and the background pure black, so colour jitter cannot move
    either across the threshold.
    """
    arr = image_tensor.numpy().transpose(1, 2, 0)
    arr = arr * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return (arr.mean(axis=2) > 0.5).astype(np.uint8)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_mask_alignment(square_image_and_mask, seed):
    """Transformed mask still lands on the transformed square (IoU > 0.95)."""
    image, mask = square_image_and_mask
    np.random.seed(seed)

    tf = get_transforms("train", img_size=224)
    out = tf(image=image, masks=[mask, np.zeros_like(mask)])

    square = _recover_square(out["image"])
    moved_mask = np.asarray(out["masks"][0]).astype(np.uint8)

    assert moved_mask.shape == square.shape, (
        f"mask shape {moved_mask.shape} != image shape {square.shape}"
    )
    # A crop can legitimately remove the square entirely; only compare when
    # something survived, and require both to survive together.
    if square.sum() == 0 and moved_mask.sum() == 0:
        pytest.skip("crop removed the square entirely for this seed")

    iou = binary_iou(square, moved_mask)
    assert iou > 0.95, (
        f"seed {seed}: mask/image IoU {iou:.4f} <= 0.95 - geometric transforms are "
        "not being applied identically to image and mask"
    )


def test_masks_survive_together():
    """Both masks are transformed, not just the first."""
    size = 256
    image = np.zeros((size, size, 3), dtype=np.uint8)
    leaf = np.zeros((size, size), dtype=np.uint8)
    lesion = np.zeros((size, size), dtype=np.uint8)
    image[60:200, 60:200] = 255
    leaf[60:200, 60:200] = 1
    lesion[100:160, 100:160] = 1  # strictly inside the leaf

    np.random.seed(0)
    out = get_transforms("train", img_size=224)(image=image, masks=[leaf, lesion])
    moved_leaf = np.asarray(out["masks"][0])
    moved_lesion = np.asarray(out["masks"][1])

    if moved_lesion.sum() == 0:
        pytest.skip("crop removed the lesion")

    outside = np.logical_and(moved_lesion > 0, moved_leaf == 0).sum()
    frac = outside / max(moved_lesion.sum(), 1)
    assert frac < 0.02, (
        f"{frac:.3%} of lesion pixels fell outside the leaf after transform - "
        "the two masks are not being moved together"
    )
