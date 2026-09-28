"""E5 - few-shot field adaptation.

Fine-tune on k real field photos per class (PlantDoc *train* split) and score on
the PlantDoc *test* split, which the model never sees. This is the arm that asks
the practical question rather than the scientific one: if you *do* have a handful
of field photos, how far does that get you?

Two axes:

* ``k`` - photos per class (5, 10, 20, or ``all``). Gives a curve, not a point:
  "how many field photos do you actually need?"
* ``init`` - start from the PlantVillage-trained checkpoint, or from plain
  ImageNet weights. If lab pretraining does not beat ImageNet here, that is a
  finding about what PlantVillage actually teaches.

Checkpoint selection is deliberately absent. There is no field validation set
large enough to select on without leaking into test, so the model after the
fixed schedule is the one reported - selecting on the 236-image test split would
inflate every number in this experiment.

This is a separate script rather than a regime in ``train/train.py`` because its
data comes from a different dataset with different splits; bolting that onto the
PlantVillage loader would complicate the loop every other experiment depends on.

Usage::

    python train/fewshot.py --config configs/E5_k20_pv_seed0.yaml
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
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
from eval.evaluate import field_label_map  # noqa: E402
from eval.metrics import accuracy_with_ci, macro_f1  # noqa: E402
from models.build import build_model  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def gather_split(root: Path, label_map: dict[str, int]) -> dict[int, list[Path]]:
    """``{label: [paths]}`` for every mapped class folder under ``root``."""
    out: dict[int, list[Path]] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if d.name not in label_map:
            continue
        imgs = sorted(p for p in d.iterdir() if p.suffix in IMAGE_EXTS and p.is_file())
        if imgs:
            out.setdefault(label_map[d.name], []).extend(imgs)
    return out


def sample_k(by_class: dict[int, list[Path]], k: int | str, seed: int):
    """k images per class, seeded. Classes with fewer than k give all they have."""
    rng = random.Random(seed)
    paths, labels, per_class = [], [], {}
    for lab in sorted(by_class):
        pool = list(by_class[lab])
        take = pool if k == "all" else rng.sample(pool, min(int(k), len(pool)))
        paths.extend(take)
        labels.extend([lab] * len(take))
        per_class[lab] = len(take)
    return paths, labels, per_class


@torch.no_grad()
def score(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for images, labels, _l, _s in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits, _ = model(images)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(labels.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def build(cfg: dict[str, Any], repo: Path, num_classes: int, device):
    """Model from a PlantVillage checkpoint or from ImageNet weights."""
    init = cfg.get("init", "pv")
    model = build_model(cfg.get("model", "resnet50"), num_classes=num_classes,
                        pretrained=(init == "imagenet"))
    if init == "pv":
        ck_path = repo / cfg["init_checkpoint"]
        if not ck_path.exists():
            raise FileNotFoundError(f"init_checkpoint not found: {ck_path}")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if int(ck["num_classes"]) != num_classes:
            raise ValueError(
                f"init checkpoint has {ck['num_classes']} classes, expected {num_classes}"
            )
        model.load_state_dict(ck["model_state"])
    elif init != "imagenet":
        raise ValueError(f"init must be 'pv' or 'imagenet', got {init!r}")
    return model.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description="E5 few-shot field adaptation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    seed = int(cfg["seed"])
    set_seed(seed, deterministic=cfg.get("deterministic", True))
    device = get_device()
    num_classes = int(cfg.get("num_classes", 38))

    label_map = field_label_map(repo, "plantdoc_name")
    pd_root = repo / cfg.get("plantdoc_root", "data/raw/plantdoc")
    train_by_class = gather_split(pd_root / "train", label_map)
    test_by_class = gather_split(pd_root / "test", label_map)

    k = cfg.get("k", 20)
    k = "all" if str(k) == "all" else int(k)
    tr_paths, tr_labels, per_class = sample_k(train_by_class, k, seed)
    te_paths = [p for lab in sorted(test_by_class) for p in test_by_class[lab]]
    te_labels = [lab for lab in sorted(test_by_class) for _ in test_by_class[lab]]

    if not tr_paths or not te_paths:
        raise ValueError(f"empty split: {len(tr_paths)} train / {len(te_paths)} test")

    img_size = int(cfg.get("img_size", 224))
    workers = int(cfg.get("num_workers", 2))
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(
        LeafDataset(tr_paths, tr_labels, transform=get_transforms("train", img_size=img_size)),
        batch_size=int(cfg.get("batch_size", 16)), shuffle=True, num_workers=workers,
        pin_memory=True, worker_init_fn=seed_worker, generator=g,
    )
    test_loader = DataLoader(
        LeafDataset(te_paths, te_labels, transform=get_transforms("test", img_size=img_size)),
        batch_size=32, shuffle=False, num_workers=0, pin_memory=True,
    )

    model = build(cfg, repo, num_classes, device)
    epochs = int(cfg.get("epochs", 15))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("lr", 1e-4)),
                            weight_decay=float(cfg.get("weight_decay", 1e-4)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    exp_id = cfg["exp_id"]
    out = run_dir(exp_id, seed)
    save_config(cfg, out / "config.yaml")
    logger = CSVLogger(out / "log.csv")

    print(f"[{exp_id} seed={seed}] init={cfg.get('init', 'pv')} k={k} "
          f"train={len(tr_paths)} ({len(per_class)} classes) test={len(te_paths)} "
          f"device={device}", flush=True)

    # Zero-shot starting point, on the same test set - the number few-shot is
    # measured against. Only meaningful for pv init; ImageNet has a random head.
    pred0, true0 = score(model, test_loader, device)
    zero_shot = accuracy_with_ci(true0, pred0, seed=seed)
    print(f"  before fine-tuning: test acc {zero_shot['point']:.4f}", flush=True)

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        tot, seen, correct = 0.0, 0, 0
        for images, labels, _l, _s in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits, _ = model(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += labels.size(0)
        sched.step()
        row = {"epoch": epoch, "train_ce": round(tot / max(seen, 1), 6),
               "train_acc": round(correct / max(seen, 1), 6), "lr": opt.param_groups[0]["lr"]}
        logger.log(row)
        print(f"  epoch {epoch}: train_ce={row['train_ce']:.4f} "
              f"train_acc={row['train_acc']:.4f}", flush=True)

    pred, true = score(model, test_loader, device)
    acc = accuracy_with_ci(true, pred, seed=seed)
    f1 = macro_f1(true, pred)

    torch.save({
        "model_state": model.state_dict(), "num_classes": num_classes,
        "model_name": cfg.get("model", "resnet50"), "exp_id": exp_id, "seed": seed,
        "val_acc": acc["point"], "config": cfg,
    }, out / "checkpoint.pt")

    metrics = {
        "exp_id": exp_id, "seed": seed, "init": cfg.get("init", "pv"), "k": k,
        "train_images": len(tr_paths), "train_per_class": per_class,
        "test_images": len(te_paths), "test_classes": len(test_by_class),
        "zero_shot_test_accuracy": zero_shot,
        "fewshot_test_accuracy": acc, "fewshot_test_macro_f1": f1,
        "gain_points": round(100 * (acc["point"] - zero_shot["point"]), 2),
        "epochs": epochs, "train_seconds": round(time.time() - t0, 1),
        "selection": "none - final epoch reported; no field val set to select on",
    }
    write_metrics(metrics, out / "metrics.json")
    print(f"\n  after fine-tuning:  test acc {acc['point']:.4f} "
          f"[{acc['lo']:.4f}, {acc['hi']:.4f}]  macro F1 {f1:.4f}", flush=True)
    print(f"  gain: {metrics['gain_points']:+.2f} points -> {out}", flush=True)


if __name__ == "__main__":
    main()
