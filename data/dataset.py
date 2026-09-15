"""Dataset and transforms.

One dataset class is used for every experiment. Masks are optional and are
returned as zero arrays when absent, so the training loop never branches on
whether masks exist - only on the configured ``regime``.

Geometric augmentation is applied to the image and both masks **jointly** via
albumentations' multi-mask support. This is load-bearing: if a random crop moved
the image but not the mask, the CAM penalty would be supervised by a
misaligned target and the whole experiment would be silently wrong.
``tests/test_mask_alignment.py`` guards this.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_IMG_SIZE = 224


def split_raw_path(image_path: str | Path) -> tuple[str, Path]:
    """Split an image path into ``(dataset_name, path_relative_to_dataset_root)``.

    Masks mirror the raw tree, i.e. an image at::

        data/raw/plantvillage/color/Tomato___healthy/x.jpg

    has its leaf mask at::

        data/masks/leaf/plantvillage/color/Tomato___healthy/x.jpg.png

    Args:
        image_path: Path containing a ``raw`` component.

    Returns:
        The dataset folder name and the path relative to it.

    Raises:
        ValueError: If the path does not sit under a ``raw`` directory.
    """
    p = Path(image_path)
    parts = p.parts
    try:
        i = len(parts) - 1 - parts[::-1].index("raw")
    except ValueError as exc:
        raise ValueError(
            f"Cannot derive a mask path for {p!s}: expected it to live under data/raw/<dataset>/"
        ) from exc
    if i + 1 >= len(parts):
        raise ValueError(f"Path {p!s} has no dataset folder under raw/")
    dataset = parts[i + 1]
    relative = Path(*parts[i + 2 :])
    return dataset, relative


def mask_path_for(image_path: str | Path, mask_root: str | Path) -> Path:
    """Path a mask for ``image_path`` would occupy under ``mask_root``.

    ``mask_root`` is the per-kind root, e.g. ``data/masks/leaf`` or
    ``data/masks/lesion_human``. The dataset folder is appended automatically.
    """
    dataset, relative = split_raw_path(image_path)
    return Path(mask_root) / dataset / f"{relative.as_posix()}.png"


def _load_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """Load a binary mask as ``uint8`` in ``{0, 1}``, or zeros if absent."""
    if not path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Mask exists but could not be read: {path}")
    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


class LeafDataset(Dataset):
    """Images with optional leaf and lesion masks.

    Args:
        paths: Image paths.
        labels: Integer class indices, parallel to ``paths``.
        leaf_mask_dir: Root of the leaf masks, e.g. ``data/masks/leaf``.
            ``None`` means leaf masks are returned as zeros.
        lesion_mask_dir: Root of the lesion masks, e.g.
            ``data/masks/lesion_human`` or ``data/masks/lesion_pseudo``.
            ``None`` means lesion masks are returned as zeros.
        transform: An albumentations ``Compose`` built by :func:`get_transforms`.

    Returns per item:
        ``(image, label, leaf_mask, lesion_mask)`` where image is
        ``3 x H x W`` float and each mask is ``H x W`` float in ``{0, 1}``.
    """

    def __init__(
        self,
        paths: Sequence[str | Path],
        labels: Sequence[int],
        leaf_mask_dir: str | Path | None = None,
        lesion_mask_dir: str | Path | None = None,
        transform: A.Compose | None = None,
    ):
        if len(paths) != len(labels):
            raise ValueError(f"paths/labels length mismatch: {len(paths)} vs {len(labels)}")
        self.paths = [Path(p) for p in paths]
        self.labels = list(labels)
        self.leaf_mask_dir = Path(leaf_mask_dir) if leaf_mask_dir else None
        self.lesion_mask_dir = Path(lesion_mask_dir) if lesion_mask_dir else None
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read image: {path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        hw = image.shape[:2]

        leaf = (
            _load_mask(mask_path_for(path, self.leaf_mask_dir), hw)
            if self.leaf_mask_dir
            else np.zeros(hw, dtype=np.uint8)
        )
        lesion = (
            _load_mask(mask_path_for(path, self.lesion_mask_dir), hw)
            if self.lesion_mask_dir
            else np.zeros(hw, dtype=np.uint8)
        )

        if self.transform is not None:
            out = self.transform(image=image, masks=[leaf, lesion])
            image_t = out["image"]
            leaf_t, lesion_t = out["masks"]
        else:
            image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            leaf_t, lesion_t = torch.from_numpy(leaf), torch.from_numpy(lesion)

        return (
            image_t,
            int(self.labels[idx]),
            torch.as_tensor(leaf_t).float(),
            torch.as_tensor(lesion_t).float(),
        )


def get_transforms(
    split: str,
    img_size: int = DEFAULT_IMG_SIZE,
    copypaste: bool = False,
    copypaste_p: float = 0.5,
    bg_bank: str | Path | None = None,
) -> A.Compose:
    """Build the albumentations pipeline for a split.

    Args:
        split: ``"train"``, ``"val"`` or ``"test"``.
        img_size: Square output resolution.
        copypaste: Enable background copy-paste augmentation (the ``copypaste``
            regime). Requires ``bg_bank``.
        copypaste_p: Probability of pasting onto a new background.
        bg_bank: Directory of background images built by
            ``counterfactual/build.py``.

    Raises:
        ValueError: On an unknown split, or copypaste without a bank.
        NotImplementedError: If copypaste is requested. It depends on leaf masks
            (Phase 2) and is implemented in Phase 3; see CLAUDE.md section 7.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split {split!r}; expected train/val/test")

    if copypaste:
        if split != "train":
            raise ValueError("copypaste is a training-time augmentation only")
        if bg_bank is None:
            raise ValueError("copypaste=True requires bg_bank")
        raise NotImplementedError(
            "copypaste lands in Phase 3 and needs leaf masks from Phase 2. "
            "Do not enable it before masks/leaf_masks.py has been run."
        )

    normalize = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]

    if split == "train":
        return A.Compose(
            [
                A.RandomResizedCrop(
                    size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.85, 1.18)
                ),
                A.HorizontalFlip(p=0.5),
                A.Affine(rotate=(-15, 15), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
                *normalize,
            ]
        )

    return A.Compose(
        [
            A.Resize(int(img_size * 1.14), int(img_size * 1.14)),
            A.CenterCrop(img_size, img_size),
            *normalize,
        ]
    )
