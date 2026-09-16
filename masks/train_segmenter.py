"""S0 - train a lesion segmenter on PlantSeg (Plan A).

Trains on the human-drawn PlantSeg masks imported by ``import_external.py``,
then ``pseudo_label.py`` runs this model over PlantVillage to produce
``lesion_pseudo/`` masks for training use only.

Defaults to tomato, which is the E2 crop and the second-largest in PlantSeg
(901 images). ``--all_crops`` widens it if the tomato subset proves too small.

**Domain shift is the thing to watch.** PlantSeg is in-the-wild field imagery;
PlantVillage is lab. This trains on the former and is applied to the latter.
Val IoU here is measured on held-out *field* images, so it is an optimistic
estimate of pseudo-label quality on lab images. That is exactly why
``verify_sheet.py`` exists and why metrics may never use pseudo masks.

Usage::

    python masks/train_segmenter.py --config configs/S0_segmenter.yaml
    python masks/train_segmenter.py --config configs/S0_segmenter.yaml --all_crops
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CSVLogger,
    get_device,
    load_config,
    save_config,
    seed_worker,
    set_seed,
    write_metrics,
)
from data.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

SPLIT_DIR = {"Training": "train", "Validation": "val", "Test": "test"}


class PlantSegLesionDataset(Dataset):
    """PlantSeg images paired with their imported binary lesion masks."""

    def __init__(self, rows: list[dict], plantseg_root: Path, mask_root: Path, transform):
        self.rows = rows
        self.plantseg_root = plantseg_root
        self.mask_root = mask_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        row = self.rows[i]
        sub = SPLIT_DIR[row["Split"]]
        img_path = self.plantseg_root / "plantsegv2" / "images" / sub / str(row["Name"])
        rel = img_path.relative_to(self.plantseg_root)
        mask_path = self.mask_root / "plantseg" / f"{rel.as_posix()}.png"

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise OSError(
                f"Lesion mask missing: {mask_path}\n"
                "Run masks/import_external.py --config configs/import_plantseg.yaml first."
            )
        mask = (m > 127).astype(np.uint8)

        out = self.transform(image=image, mask=mask)
        return out["image"], out["mask"].float().unsqueeze(0)


def get_seg_transforms(split: str, size: int = 512) -> A.Compose:
    """Segmentation transforms. Geometry is applied to image and mask jointly."""
    norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    if split == "train":
        return A.Compose(
            [
                A.LongestMaxSize(max_size=size),
                A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
                A.RandomCrop(size, size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(rotate=(-20, 20), p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
                *norm,
            ]
        )
    return A.Compose(
        [
            A.LongestMaxSize(max_size=size),
            A.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
            A.CenterCrop(size, size),
            *norm,
        ]
    )


def build_segmenter(arch: str, encoder: str = "resnet34") -> nn.Module:
    """Build the segmentation model.

    Args:
        arch: ``"unet"`` (segmentation_models_pytorch) or ``"segformer_b0"``
            (transformers). The spec allows either.
        encoder: Encoder for the U-Net, ImageNet-pretrained.

    Raises:
        ImportError: If the required library is not installed.
        ValueError: On an unknown arch.
    """
    if arch == "unet":
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for arch=unet.\n"
                "  pip install segmentation-models-pytorch"
            ) from exc
        return smp.Unet(
            encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=1
        )

    if arch == "segformer_b0":
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError(
                "transformers is required for arch=segformer_b0.\n  pip install transformers"
            ) from exc
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b0", num_labels=1, ignore_mismatched_sizes=True
        )
        return _SegformerWrapper(model)

    raise ValueError(f"Unknown arch {arch!r}; expected 'unet' or 'segformer_b0'")


class _SegformerWrapper(nn.Module):
    """Makes SegFormer's output shape match the U-Net's ``B x 1 x H x W``.

    SegFormer emits logits at 1/4 resolution, so they are upsampled back to the
    input size. Without this the loss would be computed against a mask four
    times larger than the prediction.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def dice_bce_loss(logits: Tensor, target: Tensor, eps: float = 1e-6) -> tuple[Tensor, Tensor, Tensor]:
    """Dice + BCE. Returns ``(total, bce, dice)``.

    BCE alone is dominated by the background class - lesions cover ~17% of a
    PlantSeg image on average - so Dice is what actually drives overlap.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(1, 2, 3)) + eps
    den = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = 1.0 - (num / den).mean()
    return bce + dice, bce, dice


@torch.no_grad()
def evaluate_iou(model: nn.Module, loader: DataLoader, device: torch.device, thr: float = 0.5):
    """Mean IoU and Dice over a loader, at a fixed probability threshold."""
    model.eval()
    inter_sum, union_sum, dice_sum, n = 0.0, 0.0, 0.0, 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(images)
        pred = (torch.sigmoid(logits.float()) > thr).float()
        inter = (pred * masks).sum(dim=(1, 2, 3))
        union = ((pred + masks) > 0).float().sum(dim=(1, 2, 3))
        dice = (2 * inter + 1e-6) / (pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1e-6)
        inter_sum += inter.sum().item()
        union_sum += union.sum().item()
        dice_sum += dice.sum().item()
        n += images.size(0)
    return inter_sum / max(union_sum, 1e-6), dice_sum / max(n, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the PlantSeg lesion segmenter (S0).")
    ap.add_argument("--config", required=True)
    ap.add_argument("--all_crops", action="store_true", help="train on every crop, not just tomato")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap training rows (smoke tests)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.all_crops:
        cfg["crop"] = None
    if args.limit is not None:
        cfg["limit"] = args.limit

    seed = int(cfg.get("seed", 0))
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    device = get_device()

    plantseg_root = repo / cfg.get("plantseg_root", "data/raw/plantseg")
    mask_root = repo / cfg.get("mask_root", "data/masks/lesion_human")
    meta = pd.read_csv(plantseg_root / "plantsegv2" / "Metadatav2.csv")

    crop = cfg.get("crop", "tomato")
    if crop:
        meta = meta[meta.Plant.str.lower() == str(crop).lower()]
        if meta.empty:
            raise ValueError(f"No PlantSeg rows for crop {crop!r}")

    train_rows = meta[meta.Split == "Training"].to_dict("records")
    val_rows = meta[meta.Split == "Validation"].to_dict("records")
    if cfg.get("limit"):
        train_rows = train_rows[: int(cfg["limit"])]
        val_rows = val_rows[: max(4, int(cfg["limit"]) // 4)]
    if not train_rows or not val_rows:
        raise ValueError(f"Empty split: {len(train_rows)} train / {len(val_rows)} val rows")

    size = int(cfg.get("img_size", 512))
    batch_size = int(cfg.get("batch_size", 4))
    workers = int(cfg.get("num_workers", 0))

    train_ds = PlantSegLesionDataset(
        train_rows, plantseg_root, mask_root, get_seg_transforms("train", size)
    )
    val_ds = PlantSegLesionDataset(
        val_rows, plantseg_root, mask_root, get_seg_transforms("val", size)
    )
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
    )

    arch = cfg.get("arch", "unet")
    model = build_segmenter(arch, cfg.get("encoder", "resnet34")).to(device)

    epochs = int(cfg.get("epochs", 30))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = repo / "runs" / cfg.get("exp_id", "S0_segmenter") / str(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")
    logger = CSVLogger(out_dir / "log.csv")
    ckpt_path = repo / cfg.get("out_weights", "weights/lesion_seg.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[S0] arch={arch} crop={crop or 'ALL'} train={len(train_rows)} val={len(val_rows)} "
        f"size={size} batch={batch_size} device={device} amp={use_amp}",
        flush=True,
    )

    best_iou, best_epoch, history = -1.0, -1, []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for step, (images, masks) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(images)
                loss, bce, dice = dice_bce_loss(logits.float(), masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tot += loss.item()
            nb += 1
            if step % 25 == 0:
                print(
                    f"  e{epoch} s{step}/{len(train_loader)} loss={loss.item():.4f} "
                    f"bce={bce.item():.4f} dice={dice.item():.4f}",
                    flush=True,
                )
        scheduler.step()

        val_iou, val_dice = evaluate_iou(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": round(tot / max(nb, 1), 6),
            "val_iou": round(val_iou, 6),
            "val_dice": round(val_dice, 6),
            "lr": optimizer.param_groups[0]["lr"],
        }
        logger.log(row)
        history.append(row)
        print(
            f"  epoch {epoch}: train_loss={row['train_loss']:.4f} "
            f"val_IoU={val_iou:.4f} val_Dice={val_dice:.4f}",
            flush=True,
        )

        if val_iou > best_iou:
            best_iou, best_epoch = val_iou, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": arch,
                    "encoder": cfg.get("encoder", "resnet34"),
                    "img_size": size,
                    "epoch": epoch,
                    "val_iou": val_iou,
                    "crop": crop,
                    "config": cfg,
                },
                ckpt_path,
            )

    metrics = {
        "exp_id": cfg.get("exp_id", "S0_segmenter"),
        "seed": seed,
        "arch": arch,
        "crop": crop,
        "train_images": len(train_rows),
        "val_images": len(val_rows),
        "best_val_iou": best_iou,
        "best_epoch": best_epoch,
        "train_seconds": round(time.time() - t0, 1),
        "history": history,
        "checkpoint": str(ckpt_path.relative_to(repo).as_posix()),
    }
    write_metrics(metrics, out_dir / "metrics.json")
    print(f"\nbest val IoU={best_iou:.4f} @ epoch {best_epoch} -> {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
