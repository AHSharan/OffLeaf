"""Generate every chart used in the report and slides.

Reads only saved result files (results/summary.json, results/*.json, runs/*/metrics.json),
so any figure can be regenerated later without retraining. Writes PNGs to
results/figures/ and copies them into report/Figures/.

    python scripts/make_report_figures.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "figures"
REPORT_FIGS = REPO / "report" / "Figures"
OUT.mkdir(parents=True, exist_ok=True)
REPORT_FIGS.mkdir(parents=True, exist_ok=True)

S = json.loads((REPO / "results" / "summary.json").read_text(encoding="utf-8"))

INK, MUTED, GRID = "#1f2a1c", "#6b7765", "#d9dfd3"
GOOD, BAD, ACC, NEUTRAL = "#3f6b2a", "#b0412c", "#2f5d7c", "#9aa593"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name: str) -> None:
    fig.savefig(OUT / name)
    shutil.copy(OUT / name, REPORT_FIGS / name)
    plt.close(fig)
    print("wrote", name)


def err(d):
    return [[d["point"] - d["lo"]], [d["hi"] - d["point"]]]


# ---------------------------------------------------------------- pipeline
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")

    def box(x, y, w, h, title, sub, fc):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc=fc, ec=MUTED, lw=0.9))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=10.5,
                weight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.6, color=MUTED)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1))

    lab, field, proc, model, ev = "#eef3e7", "#f6ece8", "#eef1f6", "#f3f0e2", "#e9f2ee"
    NL = chr(10)
    box(1, 32, 17, 10, "PlantVillage", NL.join(["54,305 lab images", "38 classes (training)"]), lab)
    box(1, 17, 17, 10, "PlantSeg", NL.join(["11,458 human", "lesion masks"]), field)
    box(1, 2, 17, 10, "PlantDoc", NL.join(["2,580 field images", "(testing, few-shot)"]), field)

    box(25, 32, 19, 10, "Leaf masks", NL.join(["SAM ViT-B,", "centre-point prompt"]), proc)
    box(25, 17, 19, 10, "Lesion segmenter", NL.join(["U-Net (ResNet-34),", "pseudo lesion masks"]), proc)
    box(25, 2, 19, 10, "Counterfactual set", NL.join(["same leaf,", "5 backgrounds"]), proc)

    box(51, 17, 20, 25, "Training regimes", NL.join(["baseline", "background removal",
        "attention penalty", "(OffLeaf loss)", "few-shot adaptation"]), model)

    box(78, 32, 21, 10, "Accuracy", NL.join(["lab and field,", "bootstrap 95% CI"]), ev)
    box(78, 17, 21, 10, "Attention metrics", NL.join(["off-leaf mass", "(Grad-CAM)"]), ev)
    box(78, 2, 21, 10, "Background swap", NL.join(["accuracy per cell,", "gap decomposition"]), ev)

    arrow(18, 37, 25, 37)            # PlantVillage -> leaf masks
    arrow(18, 22, 25, 22)            # PlantSeg -> segmenter
    arrow(18, 9, 25, 33)             # PlantDoc -> leaf masks
    arrow(18, 7, 25, 7)              # PlantDoc -> counterfactual set
    ax.annotate("", xy=(44.3, 9.5), xytext=(44.3, 34.5),      # leaf masks -> counterfactual
                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1.1,
                                connectionstyle="arc3,rad=-0.55"))
    arrow(44, 37, 51, 36)            # leaf masks -> training
    arrow(44, 22, 51, 24)            # pseudo lesion masks -> training
    arrow(71, 36, 78, 37)            # training -> accuracy
    arrow(71, 26, 78, 22)            # training -> attention
    arrow(71, 19, 78, 8)             # training -> background swap
    arrow(44, 6, 78, 6)              # counterfactual set -> background swap
    ax.text(50, 44.5, "OffLeaf pipeline", ha="center", fontsize=12, weight="bold", color=INK)
    save(fig, "fig_pipeline.png")


# ---------------------------------------------------------------- corner test
def fig_corner():
    c = S["corner_pixel"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    names = ["Chance\n(1 of 38)", "Noyan (2022)\nreported", "This work\n(8 corner pixels)"]
    vals = [c["chance"], c["noyan_2022_reported"], c["accuracy"]]
    bars = ax.bar(names, vals, color=[NEUTRAL, ACC, BAD], width=0.55)
    ax.errorbar(2, c["accuracy"], yerr=[[c["accuracy"] - c["lo"]], [c["hi"] - c["accuracy"]]],
                fmt="none", ecolor=INK, capsize=4, lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.1%}", ha="center", fontsize=10.5)
    ax.set_ylim(0, 0.6); ax.set_ylabel("Test accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    save(fig, "fig_corner_pixel.png")


# ---------------------------------------------------------------- collapse
def fig_collapse():
    e = S["E0"]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    names = ["PlantVillage test\n(lab)", "PlantDoc\n(field, zero-shot)"]
    d = [e["lab_test"], e["field_plantdoc"]]
    bars = ax.bar(names, [x["point"] for x in d], color=[GOOD, BAD], width=0.5)
    for i, x in enumerate(d):
        ax.errorbar(i, x["point"], yerr=err(x), fmt="none", ecolor=INK, capsize=4, lw=1)
        ax.text(i, x["point"] + 0.03, f"{x['point']:.2%}", ha="center", fontsize=11)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Accuracy")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    save(fig, "fig_collapse.png")


# ---------------------------------------------------------------- counterfactual
def fig_counterfactual():
    cf = S["E0"]["counterfactual"]
    order = [("paste_control", "Paste control\n(own background)", GOOD),
             ("lab_plain", "Lab leaf,\nlab background", GOOD),
             ("lab_field", "Lab leaf,\nfield background", BAD),
             ("field_plain", "Field leaf,\nlab background", NEUTRAL),
             ("field_field", "Field leaf,\nfield background", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = [cf[k] for k, _, _ in order]
    bars = ax.bar([n for _, n, _ in order], vals, color=[c for _, _, c in order], width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    ax.annotate("", xy=(2.0, cf["lab_field"] + 0.08), xytext=(1.3, cf["lab_plain"] - 0.03),
                arrowprops=dict(arrowstyle="-|>", color=BAD, lw=1.3))
    ax.text(2.4, 0.66, "46 points lost,\nleaf unchanged", ha="left",
            color=BAD, fontsize=10)
    ax.set_ylim(0, 1.12); ax.set_ylabel("Accuracy (200 leaves per condition)")
    save(fig, "fig_counterfactual.png")


# ---------------------------------------------------------------- training curves
def fig_training():
    m = json.loads((REPO / "results" / "E0_training_metrics.json").read_text())
    h = m["history"]
    ep = [r["epoch"] + 1 for r in h]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    a1.plot(ep, [r["train_acc"] for r in h], "-o", ms=3.5, color=ACC, label="train")
    a1.plot(ep, [r["val_acc"] for r in h], "-s", ms=3.5, color=GOOD, label="validation")
    a1.set_xlabel("Epoch"); a1.set_ylabel("Accuracy"); a1.legend(frameon=False)
    a1.set_title("Accuracy")
    a2.plot(ep, [r["train_ce"] for r in h], "-o", ms=3.5, color=ACC, label="train")
    a2.plot(ep, [r["val_loss"] for r in h], "-s", ms=3.5, color=GOOD, label="validation")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Cross-entropy loss"); a2.set_yscale("log")
    a2.legend(frameon=False); a2.set_title("Loss")
    fig.tight_layout()
    save(fig, "fig_training_E0.png")


# ---------------------------------------------------------------- epoch 3 vs 9
def fig_epochs():
    e3, e9 = S["E0_epoch3"], S["E0"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    a1.bar(["Epoch 3", "Epoch 9"], [e3["lab_val"], e9["lab_val_best"]], color=[NEUTRAL, GOOD],
           width=0.5)
    a1.set_ylim(0.99, 1.0); a1.set_title("Lab validation accuracy")
    for i, v in enumerate([e3["lab_val"], e9["lab_val_best"]]):
        a1.text(i, v + 0.0002, f"{v:.4f}", ha="center", fontsize=10)
    d = [e3["field_plantdoc"], e9["field_plantdoc"]]
    a2.bar(["Epoch 3", "Epoch 9"], [x["point"] for x in d], color=[GOOD, BAD], width=0.5)
    for i, x in enumerate(d):
        a2.errorbar(i, x["point"], yerr=err(x), fmt="none", ecolor=INK, capsize=4, lw=1)
        a2.text(i, x["hi"] + 0.01, f"{x['point']:.4f}", ha="center", fontsize=10)
    a2.set_ylim(0, 0.35); a2.set_title("Field accuracy (PlantDoc)")
    fig.tight_layout()
    save(fig, "fig_epoch_tradeoff.png")


# ---------------------------------------------------------------- E2
def fig_e2_training():
    l1, l0 = S["E2"]["lam1"], S["E2"]["lam0"]
    ep = list(range(1, 11))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.3))
    a1.plot(ep, l1["penalty_per_epoch"], "-o", ms=4, color=BAD)
    a1.set_xlabel("Epoch"); a1.set_ylabel("OffLeaf loss (off-lesion attention)")
    a1.set_title("Attention penalty during training (λ = 1)")
    a2.plot(ep, l1["val_per_epoch"], "-o", ms=3.5, color=BAD, label="λ = 1 (penalty)")
    a2.plot(ep, l0["val_per_epoch"], "-s", ms=3.5, color=ACC, label="λ = 0 (control)")
    a2.set_xlabel("Epoch"); a2.set_ylabel("Lab validation accuracy")
    a2.set_title("Lab accuracy is unaffected"); a2.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_e2_training.png")


def fig_e2_field():
    l1, l0 = S["E2"]["lam1"], S["E2"]["lam0"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, key, title in [(a1, "field", "Field accuracy (7 tomato classes)"),
                           (a2, "offleaf_mass", "Off-leaf attention on field images")]:
        d = [l0[key], l1[key]]
        ax.bar(["λ = 0", "λ = 1"], [x["point"] for x in d], color=[ACC, BAD], width=0.5)
        for i, x in enumerate(d):
            ax.errorbar(i, x["point"], yerr=err(x), fmt="none", ecolor=INK, capsize=4, lw=1)
            ax.text(i, x["hi"] + 0.01, f"{x['point']:.3f}", ha="center", fontsize=10)
        ax.set_ylim(0, 0.6); ax.set_title(title)
    fig.tight_layout()
    save(fig, "fig_e2_field.png")


# ---------------------------------------------------------------- S0
def fig_s0():
    m = json.loads((REPO / "runs" / "S0_segmenter" / "0" / "metrics.json").read_text())
    h = m["history"]
    ep = [r["epoch"] + 1 for r in h]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(ep, [r["val_iou"] for r in h], "-o", ms=3, color=GOOD, label="validation IoU")
    ax.plot(ep, [r["val_dice"] for r in h], "-s", ms=3, color=ACC, label="validation Dice")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Score"); ax.set_ylim(0.3, 0.75)
    ax.legend(frameon=False)
    save(fig, "fig_s0_segmenter.png")


# ---------------------------------------------------------------- E5
def fig_e5():
    pts = []
    for k in ["5", "10", "20", "all"]:
        p = REPO / "runs" / f"E5_k{k}_pv" / "0" / "metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            pts.append((k, m["fewshot_test_accuracy"], m["zero_shot_test_accuracy"]))
    img = REPO / "runs" / "E5_k20_imagenet" / "0" / "metrics.json"
    imnet = json.loads(img.read_text())["fewshot_test_accuracy"] if img.exists() else None
    if not pts:
        return
    zero = pts[0][2]
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    labels = ["Zero-shot"] + [("All" if k == "all" else f"k = {k}") for k, _, _ in pts]
    vals = [zero] + [a for _, a, _ in pts]
    cols = [NEUTRAL] + [GOOD] * len(pts)
    if imnet:
        labels.append("k = 20\nImageNet init"); vals.append(imnet); cols.append(ACC)
    bars = ax.bar(labels, [v["point"] for v in vals], color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax.errorbar(i, v["point"], yerr=err(v), fmt="none", ecolor=INK, capsize=3, lw=1)
        ax.text(i, v["hi"] + 0.015, f"{v['point']:.1%}", ha="center", fontsize=9.5)
    ax.set_ylim(0, 0.8); ax.set_ylabel("Accuracy on unseen PlantDoc test images")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    save(fig, "fig_e5_fewshot.png")


if __name__ == "__main__":
    for f in [fig_pipeline, fig_corner, fig_collapse, fig_counterfactual, fig_training,
              fig_epochs, fig_e2_training, fig_e2_field, fig_s0, fig_e5]:
        f()
