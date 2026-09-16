"""The single training loop. Regime is selected by config.

Regimes:

* ``baseline``    - cross-entropy only. The reference point.
* ``cam_penalty`` - CE plus ``offleaf_loss``: a GAIN-style penalty on Grad-CAM
  mass falling outside the mask. This is the intervention the project is about.
* ``copypaste``   - augmentation only; backgrounds are swapped during training
  so background pixels carry no consistent class signal.
* ``bgremoval``   - train on PlantVillage ``segmented/`` (background already
  removed). At eval the SAM leaf mask is applied to test images.

Every run writes ``runs/<exp_id>/<seed>/`` containing ``config.yaml``,
``metrics.json``, ``checkpoint.pt`` and ``log.csv``.

Usage::

    python train/train.py --config configs/E0_resnet50_seed0.yaml
    python train/train.py --config configs/E2_lam1.0_seed0.yaml --epochs 2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CSVLogger,
    get_device,
    load_config,
    run_dir,
    save_config,
    seed_worker,
    set_seed,
    write_metrics,
)
from data.dataset import LeafDataset, get_transforms  # noqa: E402
from data.splits import load_split  # noqa: E402
from models.build import build_model, normalized_cam  # noqa: E402

KNOWN_REGIMES = {"baseline", "cam_penalty", "copypaste", "bgremoval"}


def offleaf_loss(cam: Tensor, mask: Tensor) -> Tensor:
    """The OffLeaf penalty: Grad-CAM mass landing outside the mask.

    Args:
        cam: ``B x H x W`` attribution map, min-max normalised to ``[0, 1]``.
        mask: ``B x H x W`` binary mask of the region attribution *should* be on
            (leaf or lesion, per ``config.mask_type``).

    Returns:
        Scalar penalty. Zero when all attribution falls inside the mask.
    """
    return (cam * (1.0 - mask)).mean()


def select_mask(cfg: dict[str, Any], leaf: Tensor, lesion: Tensor) -> Tensor:
    """Pick the supervision mask for the CAM penalty.

    Raises:
        ValueError: On an unknown ``mask_type``, or an all-zero batch of masks,
            which means the mask directory is wrong or the masks were never
            generated. Training through that would silently penalise everything.
    """
    mask_type = cfg.get("mask_type", "leaf")
    if mask_type == "leaf":
        mask = leaf
    elif mask_type == "lesion":
        mask = lesion
    else:
        raise ValueError(f"Unknown mask_type {mask_type!r}; expected 'leaf' or 'lesion'")

    if float(mask.sum()) == 0.0:
        raise ValueError(
            f"Every {mask_type} mask in this batch is empty. Check the configured "
            f"{mask_type}_mask_dir - the masks are probably missing on disk. "
            "Training on empty masks would penalise attribution everywhere."
        )
    return mask


def variant_key(name: str) -> str:
    """Normalise a PlantVillage filename to the part that is stable across variants.

    Filenames are *not* identical between ``color/`` and ``segmented/``. Two
    differences occur, sometimes together::

        color/Apple___healthy/0055dd26-...___RS_HL 5672.JPG
        segmented/Apple___healthy/0055dd26-...___RS_HL 5672_final_masked.jpg

        color/Corn_(maize)___Common_rust_/RS_Rust 1818.JPG
        segmented/Corn_(maize)___Common_rust_/17d09699-...___RS_Rust 1818_final_masked.jpg

    i.e. a ``_final_masked`` suffix, a case change on the extension, and in ~2%
    of cases a UUID prefix present in one variant but not the other. What
    survives in every case is the original capture id after ``___``.
    """
    stem = Path(name).stem
    if "___" in stem:
        stem = stem.split("___", 1)[1]
    if stem.endswith("_final_masked"):
        stem = stem[: -len("_final_masked")]
    return stem.strip().lower()


def build_variant_index(class_dir: Path) -> dict[str, Path]:
    """Map :func:`variant_key` -> file, for one class folder of a variant."""
    index: dict[str, Path] = {}
    if not class_dir.exists():
        return index
    for p in sorted(class_dir.iterdir()):
        if p.is_file():
            index.setdefault(variant_key(p.name), p)
    return index


def resolve_paths(paths: list[Path], cfg: dict[str, Any]) -> list[Path]:
    """Apply ``image_variant`` so ``bgremoval`` trains on ``segmented/``.

    Splits are generated once against ``color/``; this maps each path onto its
    counterpart in the requested variant rather than maintaining a parallel set
    of split files, so both regimes provably see the same images in the same
    split.

    Raises:
        FileNotFoundError: If any path has no counterpart. It fails on the whole
            list rather than dropping the missing ones - silently shrinking the
            training set would make bgremoval incomparable to the regimes it is
            supposed to be measured against.
    """
    variant = cfg.get("image_variant", "color")
    if variant == "color":
        return paths

    indexes: dict[Path, dict[str, Path]] = {}
    out: list[Path] = []
    missing: list[Path] = []

    for p in paths:
        class_dir = Path(*[variant if part == "color" else part for part in p.parts[:-1]])
        if class_dir not in indexes:
            indexes[class_dir] = build_variant_index(class_dir)
        hit = indexes[class_dir].get(variant_key(p.name))
        if hit is None:
            missing.append(p)
        else:
            out.append(hit)

    if missing:
        raise FileNotFoundError(
            f"image_variant={variant!r}: {len(missing)} of {len(paths)} images have no "
            f"counterpart, e.g. {missing[0].name} in {missing[0].parent.name}. "
            f"Is data/raw/plantvillage/{variant}/ complete?"
        )
    return out


def build_loaders(cfg: dict[str, Any], repo: Path) -> tuple[DataLoader, DataLoader, int]:
    """Build train/val loaders from the split CSVs on disk."""
    seed = int(cfg["seed"])
    split_dir = repo / cfg.get("split_dir", "data/splits")
    name = cfg.get("dataset_name", "plantvillage")

    train_paths, train_labels = load_split(split_dir / f"{name}_seed{seed}_train.csv")
    val_paths, val_labels = load_split(split_dir / f"{name}_seed{seed}_val.csv")

    keep = cfg.get("keep_classes")
    if keep:
        keep = set(keep)
        pairs = [(p, y) for p, y in zip(train_paths, train_labels) if p.parent.name in keep]
        train_paths, train_labels = [p for p, _ in pairs], [y for _, y in pairs]
        pairs = [(p, y) for p, y in zip(val_paths, val_labels) if p.parent.name in keep]
        val_paths, val_labels = [p for p, _ in pairs], [y for _, y in pairs]
        # Re-index labels so the head size matches the filtered class count.
        remap = {old: i for i, old in enumerate(sorted(set(train_labels)))}
        train_labels = [remap[y] for y in train_labels]
        val_labels = [remap[y] for y in val_labels]

    train_paths = resolve_paths(train_paths, cfg)
    val_paths = resolve_paths(val_paths, cfg)
    num_classes = len(set(train_labels))

    img_size = int(cfg.get("img_size", 224))
    leaf_dir = cfg.get("leaf_mask_dir")
    lesion_dir = cfg.get("lesion_mask_dir")
    regime = cfg.get("regime", "baseline")

    train_tf = get_transforms(
        "train",
        img_size=img_size,
        copypaste=regime == "copypaste",
        copypaste_p=float(cfg.get("copypaste_p", 0.5)),
        bg_bank=(repo / cfg["bg_bank"]) if cfg.get("bg_bank") else None,
    )
    val_tf = get_transforms("val", img_size=img_size)

    train_ds = LeafDataset(
        train_paths,
        train_labels,
        leaf_mask_dir=repo / leaf_dir if leaf_dir else None,
        lesion_mask_dir=repo / lesion_dir if lesion_dir else None,
        transform=train_tf,
    )
    val_ds = LeafDataset(
        val_paths,
        val_labels,
        leaf_mask_dir=repo / leaf_dir if leaf_dir else None,
        lesion_mask_dir=repo / lesion_dir if lesion_dir else None,
        transform=val_tf,
    )

    batch_size = int(cfg.get("batch_size", 32))
    workers = int(cfg.get("num_workers", 2))
    val_workers = int(cfg.get("val_num_workers", 0))
    persistent = bool(cfg.get("persistent_workers", False))
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=g,
        persistent_workers=persistent and workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=persistent and val_workers > 0,
    )
    return train_loader, val_loader, num_classes


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return ``(loss, accuracy)`` over a loader."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for images, labels, _leaf, _lesion in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)


def compute_penalty(
    model: nn.Module,
    logits: Tensor,
    feat: Tensor,
    labels: Tensor,
    mask: Tensor,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[Tensor, Tensor]:
    """GAIN-style CAM penalty. Returns ``(penalty, cam)``.

    Follows the spec exactly: differentiate the true-class score w.r.t. the
    layer-3 features with ``create_graph=True``, form Grad-CAM, min-max
    normalise per image, upsample to mask resolution, and penalise the mass
    outside the mask.

    AMP note: the score is scaled before ``autograd.grad`` and the gradient
    unscaled afterwards, per the documented pattern for double backward under
    GradScaler. Strictly the min-max normalisation already makes the CAM
    invariant to a positive rescale, but leaving the gradient scaled by ~2**16
    risks fp16 overflow inside the CAM sum, so it is undone explicitly.
    """
    s = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    target = scaler.scale(s.sum()) if use_amp else s.sum()

    grad = torch.autograd.grad(target, feat, create_graph=True)[0]
    if use_amp:
        grad = grad / scaler.get_scale()

    cam = normalized_cam(feat.float(), grad.float(), out_hw=mask.shape[-2:])
    penalty = offleaf_loss(cam, mask)

    # Never train through a non-finite penalty. CE would keep going, validation
    # accuracy would look normal, and the intervention would quietly be absent -
    # which is indistinguishable from a null result when the numbers are read
    # months later.
    if not torch.isfinite(penalty):
        raise FloatingPointError(
            "offleaf_loss is not finite. This is the fp16 double-backward overflow: "
            "run with amp: false (the default for cam_penalty), or set penalty_amp: true "
            "only if you have verified the penalty stays finite."
        )
    return penalty, cam


def train(cfg: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Run one training job and return its metrics."""
    regime = cfg.get("regime", "baseline")
    if regime not in KNOWN_REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {sorted(KNOWN_REGIMES)}")

    seed = int(cfg["seed"])
    exp_id = cfg["exp_id"]
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    device = get_device()

    out = run_dir(exp_id, seed)
    save_config(cfg, out / "config.yaml")
    logger = CSVLogger(out / "log.csv")

    train_loader, val_loader, num_classes = build_loaders(cfg, repo)
    expected = cfg.get("num_classes")
    if expected is not None and int(expected) != num_classes:
        raise ValueError(
            f"config.num_classes={expected} but the split contains {num_classes} classes"
        )

    model = build_model(
        cfg.get("model", "resnet50"),
        num_classes=num_classes,
        pretrained=cfg.get("pretrained", True),
    ).to(device)

    optimizer_name = cfg.get("optimizer", "adamw")
    if optimizer_name != "adamw":
        raise ValueError(f"Only adamw is specified; got {optimizer_name!r}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )

    epochs = int(cfg.get("epochs", 10))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"

    lam = float(cfg.get("lam", 0.0))
    use_penalty_regime = cfg.get("regime") == "cam_penalty" and lam != 0.0

    # AMP and the CAM penalty do not mix. The penalty needs a double backward
    # (autograd.grad with create_graph=True) through the layer-3 features. Under
    # fp16 with GradScaler that overflows: measured on this repo, the penalty was
    # NaN on the first step and penalty_grad_norm was NaN on every step, while
    # the identical run in fp32 gave penalty ~0.15 and grad norms of 6.5-11.6.
    # A silently-NaN penalty is the worst failure mode available here - CE keeps
    # training, val accuracy looks fine, and the intervention does nothing.
    if use_penalty_regime and use_amp and not cfg.get("penalty_amp", False):
        print(
            "  NOTE: disabling AMP for this cam_penalty run (fp16 double backward "
            "produces NaN). Set penalty_amp: true to override. Lower batch_size if "
            "you hit CUDA OOM - fp32 plus create_graph roughly doubles activation memory.",
            flush=True,
        )
        use_amp = False
        cfg["amp"] = False

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    shuffle_masks = bool(cfg.get("shuffle_masks", False))
    log_every = int(cfg.get("log_every", 50))
    limit_batches = cfg.get("limit_batches")
    use_penalty = regime == "cam_penalty"

    print(
        f"[{exp_id} seed={seed}] regime={regime} model={cfg.get('model', 'resnet50')} "
        f"classes={num_classes} device={device} amp={use_amp}",
        flush=True,
    )
    if use_penalty:
        print(
            f"  penalty: mask_type={cfg.get('mask_type', 'leaf')} "
            f"mask_source={cfg.get('mask_source', 'n/a')} lam={lam} "
            f"shuffle_masks={shuffle_masks}",
            flush=True,
        )
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}", flush=True)

    best_acc, best_epoch = -1.0, -1
    history: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_ce, running_pen, seen, correct = 0.0, 0.0, 0, 0
        last_grad_norm = 0.0

        for step, (images, labels, leaf, lesion) in enumerate(train_loader):
            if limit_batches is not None and step >= int(limit_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits, feat = model(images)
                ce = F.cross_entropy(logits, labels)

            penalty_val = 0.0
            if use_penalty and lam != 0.0:
                mask = select_mask(cfg, leaf, lesion).to(device, non_blocking=True)
                if shuffle_masks:
                    # Control: same mask statistics, wrong image. If this scores
                    # like the real thing, the penalty is not using alignment.
                    mask = torch.roll(mask, shifts=1, dims=0)
                penalty, _cam = compute_penalty(
                    model, logits, feat, labels, mask, scaler, use_amp
                )
                loss = ce + lam * penalty
                penalty_val = float(penalty.detach())

                if step % log_every == 0:
                    g = torch.autograd.grad(
                        lam * penalty,
                        [p for p in model.parameters() if p.requires_grad],
                        retain_graph=True,
                        allow_unused=True,
                    )
                    last_grad_norm = float(
                        torch.sqrt(sum((x.float() ** 2).sum() for x in g if x is not None))
                    )
            else:
                loss = ce

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_ce += ce.item() * labels.size(0)
            running_pen += penalty_val * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

            if step % log_every == 0:
                extra = (
                    f" pen={penalty_val:.4f} pgn={last_grad_norm:.4f}" if use_penalty else ""
                )
                print(
                    f"  e{epoch} s{step}/{len(train_loader)} ce={ce.item():.4f} "
                    f"acc={correct / max(seen, 1):.4f}{extra}",
                    flush=True,
                )

        scheduler.step()
        train_loss = running_ce / max(seen, 1)
        train_acc = correct / max(seen, 1)
        train_pen = running_pen / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_ce": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
            "penalty": round(train_pen, 6),
            "penalty_grad_norm": round(last_grad_norm, 6),
        }
        logger.log(row)
        history.append(row)
        print(
            f"  epoch {epoch}: train_ce={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} penalty={train_pen:.4f}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "num_classes": num_classes,
                    "model_name": cfg.get("model", "resnet50"),
                    "exp_id": exp_id,
                    "seed": seed,
                    "config": cfg,
                },
                out / "checkpoint.pt",
            )

    metrics = {
        "exp_id": exp_id,
        "seed": seed,
        "regime": regime,
        "model": cfg.get("model", "resnet50"),
        "lam": lam,
        "mask_type": cfg.get("mask_type") if use_penalty else None,
        "mask_source": cfg.get("mask_source") if use_penalty else None,
        "shuffle_masks": shuffle_masks if use_penalty else None,
        "num_classes": num_classes,
        "epochs": epochs,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "final_penalty": history[-1]["penalty"] if history else None,
        "train_seconds": round(time.time() - t0, 1),
        "history": history,
    }
    write_metrics(metrics, out / "metrics.json")
    print(f"  best val_acc={best_acc:.4f} @ epoch {best_epoch} -> {out}", flush=True)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one OffLeaf run.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override config.seed")
    ap.add_argument("--epochs", type=int, default=None, help="override config.epochs")
    ap.add_argument("--lam", type=float, default=None, help="override config.lam")
    ap.add_argument(
        "--limit_batches", type=int, default=None, help="stop each epoch early (smoke tests)"
    )
    ap.add_argument(
        "--num_workers", type=int, default=None, help="override config.num_workers (0 = no spawn)"
    )
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)

    # Overrides are recorded into the saved config so a run stays reproducible.
    for key, val in (
        ("seed", args.seed),
        ("epochs", args.epochs),
        ("lam", args.lam),
        ("limit_batches", args.limit_batches),
        ("num_workers", args.num_workers),
    ):
        if val is not None:
            cfg[key] = val

    train(cfg, repo)


if __name__ == "__main__":
    main()
