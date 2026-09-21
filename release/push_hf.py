"""Push a checkpoint to the Hugging Face Hub with a generated model card.

The card is built from the run's own ``eval_*.json`` files, so every number on
it traces to a file on disk. Nothing is typed in by hand, which means the card
cannot drift from the results the way a manually-written one does.

Authentication is **yours, not this script's**. Log in first::

    huggingface-cli login          # or: export HF_TOKEN=...

The script never reads, prints, or stores a token.

Usage::

    python release/push_hf.py --checkpoint runs/E0_resnet50/0/checkpoint.pt \
        --repo-id <your-hf-username>/offleaf-resnet50-pv38
    python release/push_hf.py --checkpoint ... --repo-id ... --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CARD_TEMPLATE = """---
license: mit
tags:
- image-classification
- plant-disease
- agriculture
- shortcut-learning
- domain-generalization
library_name: pytorch
pipeline_tag: image-classification
---

# {title}

**Research baseline. Not deployment-ready.** See [Intended use](#intended-use).

A {model_name} trained on PlantVillage as part of **OffLeaf**, a study of how far
plant-disease classifiers rely on image *background* rather than on the disease,
and whether that reliance can be trained away.

- Repository: https://github.com/AHSharan/OffLeaf
- Training regime: `{regime}`{lam_line}
- Classes: {num_classes}
- Seed: {seed}

## The headline

{headline}

## Results

### Lab accuracy (PlantVillage held-out test)

{lab_table}

### Field accuracy (zero-shot transfer)

{field_table}

Zero-shot means the model saw **no field images during training**. This is the
setting that matters if a model is ever pointed at a real plant.

{relevance_section}

{counterfactual_section}

## Crop and class coverage

{coverage}

## Known failure modes

{failure_modes}

## Intended use

**Intended:** research on shortcut learning, background reliance and
lab-to-field generalization in plant disease classification; a baseline to
compare interventions against; a scoring target for the OffLeaf benchmark.

**Not intended:** diagnosing disease on real plants, agronomic advice, or any
decision affecting a crop. Zero-shot field accuracy is far below what a
deployed system would require, and the model has no way to say "I don't know".

A deployable system would need field training data, a leaf detector in front of
the classifier, and an abstention option. None of those are here.

## How to use

```python
import torch
from models.build import build_model      # from github.com/AHSharan/OffLeaf

ck = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
model = build_model(ck["model_name"], num_classes=ck["num_classes"], pretrained=False)
model.load_state_dict(ck["model_state"])
model.eval()

logits, features = model(images)   # features are layer-3 maps, for Grad-CAM
```

Input: 224x224 RGB, ImageNet normalisation.
Forward returns `(logits, feature_map)` - the feature map is what the
attribution metrics and the training penalty both differentiate through.

## Training

{training_details}

## Citation

If you use this model or the OffLeaf benchmark, please also cite the work it
builds on: Noyan (2022) for background bias in PlantVillage, and PlantSeg
(2026) for the lesion masks the segmenter was trained on.

---

*Card generated {generated} by `release/push_hf.py` from the run's own eval
outputs. Every number above traces to a JSON file in the repository.*
"""


def load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt_ci(d: dict | None) -> str:
    """Format a bootstrap CI dict as `0.2550 [0.2395, 0.2721]`."""
    if not d:
        return "n/a"
    return f"**{d['point']:.4f}** [{d['lo']:.4f}, {d['hi']:.4f}]"


def build_coverage(repo: Path, num_classes: int) -> str:
    cmap_path = repo / "data" / "class_map.csv"
    if not cmap_path.exists():
        return f"{num_classes} PlantVillage classes."
    cmap = pd.read_csv(cmap_path)
    crops = sorted(cmap.crop.dropna().unique())
    with_field = cmap.plantdoc_name.notna().sum()
    with_lesion = cmap.plantseg_name.notna().sum()
    lines = [
        f"- **{num_classes}** PlantVillage classes across **{len(crops)}** crops: "
        f"{', '.join(crops)}.",
        f"- **{with_field}** classes have a PlantDoc (field) counterpart - only these "
        "can be scored zero-shot.",
        f"- **{with_lesion}** classes have PlantSeg lesion-mask coverage. Classes without it "
        "cannot be pseudo-labelled, so lesion-level metrics do not apply to them.",
    ]
    return "\n".join(lines)


def build_failure_modes(field_acc: dict | None, cf: dict | None, relevance: dict | None) -> str:
    out = []
    if field_acc:
        out.append(
            f"- **Collapses on field images.** Zero-shot accuracy {field_acc['point']:.1%} "
            f"[{field_acc['lo']:.1%}, {field_acc['hi']:.1%}] versus near-ceiling in the lab. "
            "Treat any field prediction as unreliable."
        )
    if relevance and relevance.get("offleaf_mass"):
        o = relevance["offleaf_mass"]
        out.append(
            f"- **Attends to background.** {o['point']:.1%} of Grad-CAM attribution mass falls "
            "off the leaf on field images."
        )
    if cf:
        pc = cf.get("per_cell", {})
        if "lab_plain" in pc and "lab_field" in pc:
            a, b = pc["lab_plain"]["accuracy"]["point"], pc["lab_field"]["accuracy"]["point"]
            out.append(
                f"- **Prediction changes when only the background changes.** With the leaf "
                f"pixel-identical, accuracy moves {a:.1%} -> {b:.1%} when the background is "
                "swapped from lab to field."
            )
        g = cf.get("gap_decomposition", {})
        if isinstance(g, dict) and isinstance(g.get("background_share_of_gap"), (int, float)):
            out.append(
                f"- **{g['background_share_of_gap']:.0%} of the lab-to-field gap** is "
                "attributable to background rather than to the leaf."
            )
    out.append(
        "- **No abstention.** The model always returns a class, with no calibrated way to "
        "signal that an image is out of distribution."
    )
    out.append(
        "- **Class coverage is partial.** Classes absent from field datasets are untested "
        "outside the lab entirely."
    )
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Push a checkpoint + generated card to HF Hub.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--repo-id", required=True, help="e.g. username/offleaf-resnet50-pv38")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="write the card locally, push nothing")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ck_path = Path(args.checkpoint)
    if not ck_path.is_absolute():
        ck_path = repo / ck_path
    if not ck_path.exists():
        raise SystemExit(f"Checkpoint not found: {ck_path}")

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    run_dir = ck_path.parent
    cfg = ck.get("config", {})

    metrics = load_json(run_dir / "metrics.json") or {}
    field = load_json(run_dir / "eval_plantdoc.json")
    cf = load_json(run_dir / "eval_counterfactual.json")
    cf_block = cf.get("counterfactual") if cf else None

    # --- lab table -------------------------------------------------------
    lab_rows = ["| metric | value |", "|---|---|",
                f"| best val accuracy | **{metrics.get('best_val_acc', ck.get('val_acc', 0)):.4f}** |"]
    lab_test = load_json(run_dir / "eval_plantvillage_test.json")
    if lab_test:
        lab_rows.append(f"| test accuracy | {fmt_ci(lab_test.get('accuracy'))} |")
        lab_rows.append(f"| test macro F1 | {lab_test.get('macro_f1', 0):.4f} |")
    lab_table = "\n".join(lab_rows)

    # --- field table -----------------------------------------------------
    field_acc = field.get("accuracy") if field else None
    if field:
        field_table = "\n".join([
            "| dataset | accuracy | macro F1 | n |",
            "|---|---|---|---|",
            f"| PlantDoc (field) | {fmt_ci(field_acc)} | {field.get('macro_f1', 0):.4f} | "
            f"{field.get('dataset', {}).get('n', '?')} |",
        ])
    else:
        field_table = "_Not yet evaluated on a field dataset._"

    # --- relevance -------------------------------------------------------
    relevance = None
    rel_lines = []
    for key in (field or {}):
        if key.startswith("relevance_"):
            relevance = field[key]
            method = key.replace("relevance_", "")
            if relevance.get("PSEUDO_MASK_BASED"):
                rel_lines.append(
                    "> **These relevance numbers are computed on pseudo (model-generated) "
                    "masks** and are not metric-grade. Treat as indicative only."
                )
            rel_lines.append(f"**{method}**")
            rel_lines.append("")
            rel_lines.append("| metric | value |")
            rel_lines.append("|---|---|")
            if relevance.get("offleaf_mass"):
                rel_lines.append(f"| `offleaf_mass` (mass off the leaf) | "
                                 f"{fmt_ci(relevance['offleaf_mass'])} |")
            if relevance.get("offlesion_mass"):
                rel_lines.append(f"| `offlesion_mass` (mass off the lesion) | "
                                 f"{fmt_ci(relevance['offlesion_mass'])} |")
            rel_lines.append("")
    relevance_section = (
        "### Where the model looks\n\n"
        "Attribution mass outside the leaf, measured against SAM leaf masks.\n\n"
        + "\n".join(rel_lines)
    ) if rel_lines else ""

    # --- counterfactual --------------------------------------------------
    cf_section = ""
    if cf_block and cf_block.get("per_cell"):
        rows = ["### Background-swap benchmark", "",
                "Same leaf, different background. A model reading the leaf should be "
                "unaffected.", "",
                "| cell | accuracy | n |", "|---|---|---|"]
        for cell, v in cf_block["per_cell"].items():
            rows.append(f"| `{cell}` | {v['accuracy']['point']:.4f} | {v['n']} |")
        g = cf_block.get("gap_decomposition", {})
        if isinstance(g.get("background_share_of_gap"), (int, float)):
            rows += ["", f"**Background accounts for {g['background_share_of_gap']:.1%} of the "
                         "lab-to-field accuracy gap.**"]
        cf_section = "\n".join(rows)

    # --- headline --------------------------------------------------------
    best_val = metrics.get("best_val_acc", ck.get("val_acc", 0))
    if field_acc:
        headline = (
            f"This model scores **{best_val:.1%}** on PlantVillage and "
            f"**{field_acc['point']:.1%}** on real field photographs of the same diseases. "
            "That gap, not the lab number, is what the OffLeaf project measures."
        )
    else:
        headline = f"Lab accuracy **{best_val:.1%}**. Field evaluation pending."

    lam = cfg.get("lam")
    card = CARD_TEMPLATE.format(
        title=f"OffLeaf {ck.get('model_name', 'model')} ({cfg.get('regime', 'baseline')})",
        model_name=ck.get("model_name", "model"),
        regime=cfg.get("regime", "baseline"),
        lam_line=f"\n- Penalty strength lambda: `{lam}`" if lam else "",
        num_classes=ck.get("num_classes", "?"),
        seed=ck.get("seed", 0),
        headline=headline,
        lab_table=lab_table,
        field_table=field_table,
        relevance_section=relevance_section,
        counterfactual_section=cf_section,
        coverage=build_coverage(repo, ck.get("num_classes", 0)),
        failure_modes=build_failure_modes(field_acc, cf_block, relevance),
        training_details="\n".join([
            f"- Backbone: `{ck.get('model_name')}`, ImageNet-pretrained, single linear head",
            f"- Optimiser: {cfg.get('optimizer', 'adamw')}, lr {cfg.get('lr')}, "
            f"weight decay {cfg.get('weight_decay')}",
            f"- {cfg.get('epochs')} epochs, batch {cfg.get('batch_size')}, "
            f"{cfg.get('img_size', 224)}px, cosine schedule",
            f"- Mixed precision: {cfg.get('amp')}",
            f"- Training time: {metrics.get('train_seconds', 0) / 60:.1f} min",
        ]),
        generated=date.today().isoformat(),
    )

    card_path = run_dir / "README.md"
    card_path.write_text(card, encoding="utf-8")
    print(f"card written: {card_path}\n")

    if args.dry_run:
        print("--dry-run: nothing pushed.")
        print(card[:1500] + "\n...")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("pip install huggingface-hub") from exc

    api = HfApi()  # reads your own login; this script never handles a token
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    print(f"uploading {ck_path.name} ({ck_path.stat().st_size / 1e6:.0f} MB)...", flush=True)
    api.upload_file(path_or_fileobj=str(ck_path), path_in_repo="checkpoint.pt",
                    repo_id=args.repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="model")
    print(f"\ndone: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
