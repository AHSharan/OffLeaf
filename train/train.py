"""The single training loop. Regime is selected by config.

Phase 1 implements ``baseline`` only. ``cam_penalty``, ``copypaste`` and
``bgremoval`` are Phase 3 and depend on masks from Phase 2; they raise rather
than silently training something that is not what the config asked for.

Every run writes ``runs/<exp_id>/<seed>/`` containing ``config.yaml``,
``metrics.json``, ``checkpoint.pt`` and ``log.csv``.

Usage::

    python train/train.py --config configs/E0_resnet50_seed0.yaml
    python train/train.py --config configs/E0_resnet50_seed0.yaml --epochs 2   # smoke test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
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
from models.build import build_model  # noqa: E402

IMPLEMENTED_REGIMES = {"baseline"}
KNOWN_REGIMES = {"baseline", "cam_penalty", "copypaste", "bgremoval"}


def build_loaders(cfg: dict[str, Any], repo: Path) -> tuple[DataLoader, DataLoader, int]:
    """Build train/val loaders from the split CSVs on disk."""
    seed = int(cfg["seed"])
    split_dir = repo / cfg.get("split_dir", "data/splits")
    name = cfg.get("dataset_name", "plantvillage")

    train_paths, train_labels = load_split(split_dir / f"{name}_seed{seed}_train.csv")
    val_paths, val_labels = load_split(split_dir / f"{name}_seed{seed}_val.csv")
    num_classes = len(set(train_labels))

    img_size = int(cfg.get("img_size", 224))
    leaf_dir = cfg.get("leaf_mask_dir")
    lesion_dir = cfg.get("lesion_mask_dir")

    train_ds = LeafDataset(
        train_paths,
        train_labels,
        leaf_mask_dir=repo / leaf_dir if leaf_dir else None,
        lesion_mask_dir=repo / lesion_dir if lesion_dir else None,
        transform=get_transforms("train", img_size=img_size),
    )
    val_ds = LeafDataset(
        val_paths,
        val_labels,
        leaf_mask_dir=repo / leaf_dir if leaf_dir else None,
        lesion_mask_dir=repo / lesion_dir if lesion_dir else None,
        transform=get_transforms("val", img_size=img_size),
    )

    batch_size = int(cfg.get("batch_size", 32))
    workers = int(cfg.get("num_workers", 2))
    # Validation gets its own worker count, defaulting to 0. On Windows workers
    # are spawned, not forked, so each re-imports torch and costs ~1 GB of
    # commit. With persistent train workers alive during validation the peak is
    # doubled, which is enough to exhaust commit on a 16 GB machine and kill the
    # run with "bad allocation" or a bare SystemError during a worker import.
    # Validation is infrequent and short, so 0 costs little.
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


def train(cfg: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Run one training job and return its metrics."""
    regime = cfg.get("regime", "baseline")
    if regime not in KNOWN_REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {sorted(KNOWN_REGIMES)}")
    if regime not in IMPLEMENTED_REGIMES:
        raise NotImplementedError(
            f"regime {regime!r} lands in Phase 3 and needs masks from Phase 2. "
            f"Implemented so far: {sorted(IMPLEMENTED_REGIMES)}. See CLAUDE.md section 7."
        )

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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    log_every = int(cfg.get("log_every", 50))
    limit_batches = cfg.get("limit_batches")

    print(
        f"[{exp_id} seed={seed}] regime={regime} model={cfg.get('model', 'resnet50')} "
        f"classes={num_classes} device={device} amp={use_amp}"
    )
    print(f"  train batches={len(train_loader)}  val batches={len(val_loader)}")

    best_acc, best_epoch = -1.0, -1
    history: list[dict[str, Any]] = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_ce, seen, correct = 0.0, 0, 0

        for step, (images, labels, _leaf, _lesion) in enumerate(train_loader):
            if limit_batches is not None and step >= int(limit_batches):
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits, _feat3 = model(images)
                ce = F.cross_entropy(logits, labels)
                loss = ce  # baseline: no penalty term

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_ce += ce.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)

            if step % log_every == 0:
                # flush: these runs are long and usually backgrounded, where
                # Python would otherwise buffer progress into invisibility.
                print(
                    f"  e{epoch} s{step}/{len(train_loader)} "
                    f"ce={ce.item():.4f} acc={correct / max(seen, 1):.4f}",
                    flush=True,
                )

        scheduler.step()
        train_loss, train_acc = running_ce / max(seen, 1), correct / max(seen, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_ce": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "lr": optimizer.param_groups[0]["lr"],
            # Present but zero for baseline, so every regime shares one schema.
            "penalty": 0.0,
            "penalty_grad_norm": 0.0,
        }
        logger.log(row)
        history.append(row)
        print(
            f"  epoch {epoch}: train_ce={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}",
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
        "num_classes": num_classes,
        "epochs": epochs,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "train_seconds": round(time.time() - t0, 1),
        "history": history,
    }
    write_metrics(metrics, out / "metrics.json")
    print(f"  best val_acc={best_acc:.4f} @ epoch {best_epoch} -> {out}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(description="Train one OffLeaf run.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None, help="override config.seed")
    ap.add_argument("--epochs", type=int, default=None, help="override config.epochs")
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
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.limit_batches is not None:
        cfg["limit_batches"] = args.limit_batches
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    train(cfg, repo)


if __name__ == "__main__":
    main()
