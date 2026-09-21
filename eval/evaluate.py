"""Score a checkpoint (or any timm model) on lab, field, or counterfactual data.

Usage::

    # lab test split
    python eval/evaluate.py --checkpoint runs/E0_resnet50/0/checkpoint.pt --dataset plantvillage_test

    # zero-shot field transfer, with attribution metrics
    python eval/evaluate.py --checkpoint runs/E0_resnet50/0/checkpoint.pt \
        --dataset plantdoc --explain gradcam,hirescam

    # the background-swap benchmark
    python eval/evaluate.py --checkpoint runs/E0_resnet50/0/checkpoint.pt --counterfactual

Field datasets use different class names from PlantVillage, so labels are mapped
through ``data/class_map.csv``. Field classes with no PlantVillage counterpart
are **dropped and counted**, never silently scored as wrong - a model cannot be
penalised for failing to predict a class it was never given.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device, set_seed  # noqa: E402
from data.dataset import LeafDataset, get_transforms, mask_path_for  # noqa: E402
from data.splits import load_split  # noqa: E402
from eval.metrics import (  # noqa: E402
    accuracy,
    accuracy_with_ci,
    flip_rate,
    gap_decomposition,
    macro_f1,
    per_class_report,
    relevance_mass,
)
from explain.methods import METHODS, explain, predict  # noqa: E402
from models.build import build_model  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
CELLS = ("lab_plain", "lab_field", "field_plain", "field_field", "paste_control")


# --------------------------------------------------------------------------
# model + label space
# --------------------------------------------------------------------------


def load_model(repo: Path, checkpoint: str | None, timm_model: str | None, device):
    """Load a repo checkpoint or a bare timm model. Returns ``(model, meta)``."""
    if checkpoint:
        ck = torch.load(repo / checkpoint if not Path(checkpoint).is_absolute() else checkpoint,
                        map_location="cpu", weights_only=False)
        model = build_model(ck["model_name"], num_classes=ck["num_classes"], pretrained=False)
        model.load_state_dict(ck["model_state"])
        meta = {
            "source": "checkpoint",
            "path": str(checkpoint),
            "model": ck["model_name"],
            "num_classes": ck["num_classes"],
            "exp_id": ck.get("exp_id"),
            "seed": ck.get("seed"),
            "regime": ck.get("config", {}).get("regime"),
            "lam": ck.get("config", {}).get("lam"),
            "val_acc": ck.get("val_acc"),
        }
    elif timm_model:
        raise SystemExit(
            "--timm_model needs --num_classes and a head trained for this label space. "
            "Score a repo checkpoint instead, or fine-tune the timm model first."
        )
    else:
        raise SystemExit("Pass --checkpoint")
    return model.to(device).eval(), meta


def class_index(repo: Path, dataset_name: str = "plantvillage") -> dict[str, int]:
    """PlantVillage class name -> label index, as the splits defined it."""
    p = repo / "data" / "splits" / f"{dataset_name}_classes.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run data/splits.py first.")
    df = pd.read_csv(p)
    return {str(r.class_name): int(r.label) for r in df.itertuples()}


def field_label_map(repo: Path, column: str) -> dict[str, int]:
    """Field dataset folder name -> PlantVillage label index, via class_map.csv."""
    cmap = pd.read_csv(repo / "data" / "class_map.csv")
    pv_idx = class_index(repo)
    out: dict[str, int] = {}
    for row in cmap.to_dict("records"):
        field_name, pv_name = row.get(column), row.get("plantvillage_name")
        if isinstance(field_name, str) and field_name.strip() and isinstance(pv_name, str):
            if pv_name in pv_idx:
                out[field_name.strip()] = pv_idx[pv_name]
    return out


# --------------------------------------------------------------------------
# dataset assembly
# --------------------------------------------------------------------------


def gather_folder(root: Path, label_map: dict[str, int] | None):
    """Collect ``(paths, labels)`` from a ``<root>/<class>/*.jpg`` tree."""
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    paths, labels, dropped = [], [], {}
    for d in sorted(p for p in root.rglob("*") if p.is_dir()):
        imgs = [p for p in sorted(d.iterdir()) if p.suffix in IMAGE_EXTS and p.is_file()]
        if not imgs:
            continue
        if label_map is None:
            continue
        if d.name not in label_map:
            dropped[d.name] = len(imgs)
            continue
        paths.extend(imgs)
        labels.extend([label_map[d.name]] * len(imgs))
    return paths, labels, dropped


def build_eval_set(repo: Path, dataset: str, img_size: int):
    """Returns ``(paths, labels, dropped, description)``."""
    if dataset == "plantvillage_test":
        p, y = load_split(repo / "data" / "splits" / "plantvillage_seed0_test.csv")
        return p, y, {}, "PlantVillage held-out test split (lab)"

    if dataset == "plantdoc":
        lm = field_label_map(repo, "plantdoc_name")
        p, y, dropped = gather_folder(repo / "data" / "raw" / "plantdoc", lm)
        return p, y, dropped, "PlantDoc (field, zero-shot)"

    root = Path(dataset)
    if not root.is_absolute():
        root = repo / dataset
    lm = class_index(repo)
    p, y, dropped = gather_folder(root, lm)
    return p, y, dropped, f"folder: {root}"


def make_loader(paths, labels, repo, img_size, batch_size, leaf_dir=None, lesion_dir=None):
    ds = LeafDataset(
        paths,
        labels,
        leaf_mask_dir=leaf_dir,
        lesion_mask_dir=lesion_dir,
        transform=get_transforms("test", img_size=img_size),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


# --------------------------------------------------------------------------
# core evaluation
# --------------------------------------------------------------------------


def run_eval(model, loader, device, methods: list[str], want_masks: bool,
             apply_leaf_mask: bool = False):
    """One pass: predictions, confidences, and optionally attribution maps.

    ``apply_leaf_mask`` zeroes everything outside the leaf before the forward
    pass. It is **required** when scoring a ``bgremoval`` checkpoint: that model
    trained only on leaves against a blank field, so showing it a full scene is
    maximally out of distribution and measures the wrong thing. The spec calls
    for it directly ("at eval apply the SAM leaf mask to test images").
    """
    preds, confs, trues = [], [], []
    heat: dict[str, list] = {m: [] for m in methods}
    leaves, lesions = [], []

    for images, labels, leaf, lesion in loader:
        images = images.to(device, non_blocking=True)
        if apply_leaf_mask:
            m = leaf.to(device, non_blocking=True).unsqueeze(1)
            if m.sum() == 0:
                raise ValueError(
                    "--apply_leaf_mask was passed but every leaf mask in this batch is "
                    "empty. Generate leaf masks for this dataset first "
                    "(masks/leaf_masks.py), or the model is being shown blank images."
                )
            images = images * m
        p, c = predict(model, images)
        preds.append(p.cpu().numpy())
        confs.append(c.cpu().numpy())
        trues.append(labels.numpy())

        if methods:
            for m in methods:
                cam = explain(model, images, labels.to(device), method=m)
                heat[m].append(cam.cpu().numpy())
        if want_masks:
            leaves.append(leaf.numpy())
            lesions.append(lesion.numpy())

    out = {
        "pred": np.concatenate(preds),
        "conf": np.concatenate(confs),
        "true": np.concatenate(trues),
    }
    for m in methods:
        out[f"heat_{m}"] = np.concatenate(heat[m]) if heat[m] else None
    if want_masks:
        out["leaf"] = np.concatenate(leaves)
        out["lesion"] = np.concatenate(lesions)
    return out


def counterfactual_eval(repo, model, device, img_size, batch_size, out_dir: Path) -> dict:
    """Score every cell of the background-swap benchmark and decompose the gap."""
    cf_root = repo / "data" / "counterfactual"
    index = cf_root / "index.csv"
    if not index.exists():
        raise FileNotFoundError(
            f"{index} not found. Build it: python counterfactual/build.py "
            "--config configs/counterfactual_tomato.yaml"
        )
    df = pd.read_csv(index)

    # Labels in index.csv are the SOURCE dataset's folder names, so field cells
    # carry PlantDoc names ("Tomato Early blight leaf") which are absent from the
    # PlantVillage label space. Mapping per source dataset is required or the two
    # field cells silently evaluate as empty - which is exactly what happened.
    label_maps = {
        "plantvillage": class_index(repo),
        "plantdoc": field_label_map(repo, "plantdoc_name"),
    }

    cell_acc, per_cell, by_id, unmapped = {}, {}, {}, {}
    for cell in CELLS:
        sub = df[df.cell == cell]
        if sub.empty:
            continue
        paths, labels, ids = [], [], []
        for r in sub.to_dict("records"):
            lmap = label_maps.get(str(r["source_dataset"]), {})
            lab = lmap.get(str(r["label"]))
            if lab is None:
                unmapped[str(r["label"])] = unmapped.get(str(r["label"]), 0) + 1
                continue
            fp = cf_root / cell / f"{r['image_id']}.jpg"
            if not fp.exists():
                continue
            paths.append(fp)
            labels.append(lab)
            ids.append(r["image_id"])
        if not paths:
            continue

        res = run_eval(model, make_loader(paths, labels, repo, img_size, batch_size), device, [], False)
        cell_acc[cell] = accuracy(res["true"], res["pred"])
        per_cell[cell] = {
            "n": len(paths),
            "accuracy": accuracy_with_ci(res["true"], res["pred"]),
            "macro_f1": macro_f1(res["true"], res["pred"]),
        }
        by_id[cell] = {i: (int(p), float(c)) for i, p, c in zip(ids, res["pred"], res["conf"])}

    # Flip rate: same leaf, different background. Compared against lab_plain.
    flips = {}
    if "lab_plain" in by_id:
        base = by_id["lab_plain"]
        for cell in ("lab_field", "paste_control"):
            if cell not in by_id:
                continue
            shared = sorted(set(base) & set(by_id[cell]))
            if not shared:
                continue
            flips[f"lab_plain_vs_{cell}"] = flip_rate(
                [base[i][0] for i in shared],
                [base[i][1] for i in shared],
                [by_id[cell][i][0] for i in shared],
                [by_id[cell][i][1] for i in shared],
            )

    out = {"per_cell": per_cell, "flip_rates": flips, "unmapped_labels": unmapped}
    if unmapped:
        print(
            f"WARNING: {sum(unmapped.values())} composite(s) had labels with no PlantVillage "
            f"counterpart and were skipped: {sorted(unmapped)[:5]}",
            flush=True,
        )
    try:
        out["gap_decomposition"] = gap_decomposition(cell_acc)
    except KeyError as e:
        out["gap_decomposition"] = {"error": str(e), "cells_present": sorted(cell_acc)}
    return out


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model.")
    ap.add_argument("--checkpoint")
    ap.add_argument("--timm_model")
    ap.add_argument("--dataset", default="plantvillage_test")
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--explain", default="", help="comma list, e.g. gradcam,hirescam")
    ap.add_argument("--leaf_mask_dir", default="data/masks/leaf")
    ap.add_argument("--lesion_mask_dir", default="data/masks/lesion_human")
    ap.add_argument("--allow_pseudo", action="store_true")
    ap.add_argument(
        "--apply_leaf_mask",
        action="store_true",
        help="mask test images to the leaf before inference; REQUIRED for bgremoval "
             "checkpoints (auto-enabled when the checkpoint says regime=bgremoval)",
    )
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    set_seed(args.seed)
    device = get_device()

    methods = [m.strip() for m in args.explain.split(",") if m.strip()]
    for m in methods:
        if m not in METHODS:
            raise SystemExit(f"Unknown explainer {m!r}; available: {METHODS}")

    model, meta = load_model(repo, args.checkpoint, args.timm_model, device)
    print(f"model: {meta}", flush=True)

    out_dir = Path(args.out) if args.out else (
        (repo / args.checkpoint).parent if args.checkpoint else repo / "runs" / "eval"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"model": meta, "seed": args.seed}

    if args.counterfactual:
        print("evaluating counterfactual benchmark...", flush=True)
        result["counterfactual"] = counterfactual_eval(
            repo, model, device, args.img_size, args.batch_size, out_dir
        )
        tag = "counterfactual"
    else:
        paths, labels, dropped, desc = build_eval_set(repo, args.dataset, args.img_size)
        if args.limit:
            paths, labels = paths[: args.limit], labels[: args.limit]
        if not paths:
            raise SystemExit(f"No images to evaluate for --dataset {args.dataset!r}")
        print(f"{desc}: {len(paths)} images, {len(set(labels))} classes", flush=True)
        if dropped:
            print(f"  dropped {sum(dropped.values())} images in {len(dropped)} unmapped classes:",
                  flush=True)
            for k, v in sorted(dropped.items()):
                print(f"    {v:5d}  {k}", flush=True)

        apply_mask = args.apply_leaf_mask
        if meta.get("regime") == "bgremoval" and not apply_mask:
            print(
                "NOTE: checkpoint regime is bgremoval - auto-enabling --apply_leaf_mask. "
                "Scoring it on unmasked scenes would measure out-of-distribution "
                "behaviour, not the intervention.",
                flush=True,
            )
            apply_mask = True
        want_masks = bool(methods) or apply_mask
        loader = make_loader(
            paths, labels, repo, args.img_size, args.batch_size,
            leaf_dir=repo / args.leaf_mask_dir if want_masks else None,
            lesion_dir=repo / args.lesion_mask_dir if want_masks else None,
        )
        res = run_eval(model, loader, device, methods, want_masks, apply_leaf_mask=apply_mask)

        result["dataset"] = {"name": args.dataset, "description": desc, "n": len(paths),
                             "dropped_classes": dropped,
                             "leaf_mask_applied_at_eval": bool(apply_mask)}
        result["accuracy"] = accuracy_with_ci(res["true"], res["pred"], seed=args.seed)
        result["macro_f1"] = macro_f1(res["true"], res["pred"])
        result["mean_confidence"] = float(res["conf"].mean())
        result["per_class"] = per_class_report(res["true"], res["pred"])

        for m in methods:
            heat = res[f"heat_{m}"]
            has_lesion = res["lesion"].sum() > 0
            result[f"relevance_{m}"] = relevance_mass(
                list(heat),
                list(res["leaf"]),
                list(res["lesion"]) if has_lesion else None,
                lesion_mask_dir=args.lesion_mask_dir if has_lesion else None,
                allow_pseudo=args.allow_pseudo,
                seed=args.seed,
            )
            if not has_lesion:
                result[f"relevance_{m}"]["note"] = (
                    "No lesion masks found for these images - leaf-level offleaf_mass only."
                )

        with open(out_dir / f"predictions_{args.dataset.replace('/', '_')}.csv", "w",
                  newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "true", "pred", "confidence", "correct"])
            for p, t, pr, c in zip(paths, res["true"], res["pred"], res["conf"]):
                w.writerow([Path(p).as_posix(), int(t), int(pr), round(float(c), 4), int(t == pr)])
        tag = args.dataset.replace("/", "_")

    path = out_dir / f"eval_{tag}.json"
    path.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")

    print("\n=== summary ===", flush=True)
    if "accuracy" in result:
        a = result["accuracy"]
        print(f"accuracy  {a['point']:.4f}  [{a['lo']:.4f}, {a['hi']:.4f}]  n={a['n']}")
        print(f"macro F1  {result['macro_f1']:.4f}")
        for m in methods:
            r = result[f"relevance_{m}"]
            if r.get("offleaf_mass"):
                o = r["offleaf_mass"]
                print(f"{m}: offleaf_mass {o['point']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}]")
            if r.get("offlesion_mass"):
                o = r["offlesion_mass"]
                print(f"{m}: offlesion_mass {o['point']:.4f} [{o['lo']:.4f}, {o['hi']:.4f}]")
    if "counterfactual" in result:
        for cell, v in result["counterfactual"]["per_cell"].items():
            print(f"{cell:15s} acc {v['accuracy']['point']:.4f}  n={v['n']}")
        g = result["counterfactual"].get("gap_decomposition", {})
        if "background_share_of_gap" in g:
            print(f"\nbackground share of lab-to-field gap: {g['background_share_of_gap']:.1%}")
    print(f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
