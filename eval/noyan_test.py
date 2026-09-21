"""The corner-pixel test: can background alone classify PlantVillage?

Replicates Noyan (2022), "Uncovering bias in the PlantVillage dataset", which
trained on 8 background pixels and reached 49.0% on a 38-class problem where
chance is 2.6%.

This is a **replication**, not a contribution - cite Noyan. Its job here is to
establish, on our exact splits, that the shortcut the rest of the project
measures is present in the data at all. It uses no neural network and no GPU:
if a logistic regression on 8 pixels beats chance by an order of magnitude, the
leak is in the dataset, not in any particular model.

Usage::

    python eval/noyan_test.py --config configs/noyan.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, set_seed  # noqa: E402
from data.splits import load_split  # noqa: E402
from eval.metrics import accuracy_with_ci, macro_f1  # noqa: E402

# Eight pixels: the four corners, each sampled at two insets. Insets avoid the
# very edge, where JPEG ringing and any border artefact would be an even more
# trivial giveaway than the background colour itself.
INSETS = (2, 6)


def corner_features(path: Path) -> np.ndarray | None:
    """24 features: 8 corner pixels x RGB. ``None`` if unreadable or too small."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    if h < 2 * max(INSETS) + 1 or w < 2 * max(INSETS) + 1:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    px = []
    for inset in INSETS:
        px.extend([
            rgb[inset, inset],
            rgb[inset, w - 1 - inset],
            rgb[h - 1 - inset, inset],
            rgb[h - 1 - inset, w - 1 - inset],
        ])
    return np.concatenate(px)


def build_matrix(paths, labels, limit=None):
    X, y, skipped = [], [], 0
    for i, (p, lab) in enumerate(zip(paths, labels)):
        if limit and len(y) >= limit:
            break
        f = corner_features(Path(p))
        if f is None:
            skipped += 1
            continue
        X.append(f)
        y.append(lab)
    return np.asarray(X), np.asarray(y), skipped


def main() -> None:
    ap = argparse.ArgumentParser(description="Corner-pixel background leakage test.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset_name", default="plantvillage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="cap train images (speed)")
    ap.add_argument("--classifier", choices=["logreg", "rf"], default="rf")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config) if args.config else {}
    seed = int(cfg.get("seed", args.seed))
    name = cfg.get("dataset_name", args.dataset_name)
    set_seed(seed)

    split_dir = repo / cfg.get("split_dir", "data/splits")
    tr_p, tr_y = load_split(split_dir / f"{name}_seed{seed}_train.csv")
    te_p, te_y = load_split(split_dir / f"{name}_seed{seed}_test.csv")

    print(f"extracting corner pixels from {len(tr_p)} train / {len(te_p)} test images...",
          flush=True)
    Xtr, ytr, sk_tr = build_matrix(tr_p, tr_y, args.limit)
    Xte, yte, sk_te = build_matrix(te_p, te_y)
    print(f"  train {Xtr.shape}, test {Xte.shape} (skipped {sk_tr}/{sk_te})", flush=True)

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    if args.classifier == "logreg":
        clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class="multinomial")
    else:
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)
    print(f"fitting {args.classifier} on {Xtr.shape[1]} features...", flush=True)
    clf.fit(Xtr_s, ytr)

    pred = clf.predict(Xte_s)
    n_classes = int(len(set(ytr.tolist())))
    chance = 1.0 / n_classes
    acc = accuracy_with_ci(yte, pred, seed=seed)

    result = {
        "replication_of": "Noyan (2022), Uncovering bias in the PlantVillage dataset",
        "dataset": name,
        "seed": seed,
        "classifier": args.classifier,
        "n_features": int(Xtr.shape[1]),
        "n_pixels": 8,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "n_classes": n_classes,
        "chance_accuracy": chance,
        "accuracy": acc,
        "macro_f1": macro_f1(yte, pred),
        "times_chance": acc["point"] / chance,
    }

    out = Path(args.out) if args.out else repo / "runs" / "noyan_test" / f"seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== corner-pixel test ===")
    print(f"  8 background pixels, {n_classes} classes")
    print(f"  accuracy {acc['point']:.4f}  [{acc['lo']:.4f}, {acc['hi']:.4f}]")
    print(f"  chance   {chance:.4f}")
    print(f"  ratio    {result['times_chance']:.1f}x chance")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
