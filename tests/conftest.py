"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def square_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """A white square on black, plus the matching mask.

    Deliberately high-contrast: after ImageNet normalisation the square is
    strongly positive and the background strongly negative, so the square can be
    recovered from the transformed tensor by thresholding at zero. That is what
    lets the alignment test compare geometry without inverting the augmentation.
    """
    size, half = 256, 60
    image = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    image[c - half : c + half, c - half : c + half] = 255
    mask[c - half : c + half, c - half : c + half] = 1
    return image, mask


def binary_iou(a: np.ndarray, b: np.ndarray) -> float:
    """IoU of two boolean arrays; 1.0 when both are empty."""
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)
