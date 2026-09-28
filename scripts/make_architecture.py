"""Architecture diagrams for the report and slides, in the style of the guide's
reference figures: dashed group panels, dotted sub-panels, filled cards with
bullets, and real dataset thumbnails.

Writes results/figures/fig_arch_system.png and fig_arch_workflow.png, and copies
them to report/Figures/ when that folder exists.
"""
import random
import shutil
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "figures"
REPORT_FIGS = REPO / "report" / "Figures"
plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "savefig.dpi": 220, "savefig.bbox": "tight"})
random.seed(7)

BLUE = "#d9e8f6"
BEIGE = "#f4ead0"
GREY = "#f2f2f2"
EDGE = "#555555"
NL = chr(10)


# ---------------------------------------------------------------- images
def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def square(img, size=220):
    h, w = img.shape[:2]
    s = min(h, w)
    y, x = (h - s) // 2, (w - s) // 2
    return cv2.resize(img[y:y + s, x:x + s], (size, size))


def thumbs():
    pv = REPO / "data/raw/plantvillage/color/Tomato___Early_blight"
    pv_img = square(rgb(sorted(pv.glob("*"))[40]))
    pd_img = square(rgb(sorted((REPO / "data/raw/plantdoc/train/Tomato Early blight leaf").glob("*"))[3]))

    ps = REPO / "data/raw/plantseg/plantsegv2"
    name = "tomato_bacterial_leaf_spot_14"
    img = rgb(next((ps / "images/test").glob(name + ".*")))
    ann = cv2.imread(str(ps / "annotations/test" / (name + ".png")), cv2.IMREAD_GRAYSCALE)
    ann = cv2.resize(ann, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ov = img.copy()
    ov[ann] = (0.45 * ov[ann] + 0.55 * np.array([220, 40, 40])).astype(np.uint8)
    ps_img = square(ov)

    # leaf mask overlay on the same PlantVillage image
    src = sorted(pv.glob("*"))[40]
    leaf_p = REPO / "data/masks/leaf/plantvillage/color/Tomato___Early_blight" / (src.name + ".png")
    base = rgb(src)
    m = cv2.imread(str(leaf_p), cv2.IMREAD_GRAYSCALE) > 0
    m = cv2.resize(m.astype(np.uint8), (base.shape[1], base.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ov = base.copy()
    ov[~m] = (0.3 * ov[~m]).astype(np.uint8)
    ov[m] = (0.55 * ov[m] + 0.45 * np.array([60, 200, 60])).astype(np.uint8)
    edge = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    ov[edge] = (255, 230, 0)
    mask_img = square(ov)

    cf = sorted((REPO / "data/counterfactual/lab_field").glob("*.jpg"))
    cf_img = square(rgb(cf[5])) if cf else pd_img

    demo = rgb(REPO / "report/Figures/demo_raw.webp") if (REPO / "report/Figures/demo_raw.webp").exists() else None
    if demo is not None:
        h, w = demo.shape[:2]
        cam = demo[int(0.24 * h):int(0.64 * h), int(0.6 * w):int(0.9 * w)]
        cam_img = square(cam)
    else:
        cam_img = pv_img
    return dict(pv=pv_img, pd=pd_img, ps=ps_img, mask=mask_img, cf=cf_img, cam=cam_img)


# ---------------------------------------------------------------- drawing
def panel(ax, x, y, w, h, label=None, ls=(0, (6, 4)), lw=1.6, pad=0.25, fs=13, label_y=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                                fill=False, ec="#333333", lw=lw, ls=ls))
    if label:
        ax.text(x + w / 2, label_y if label_y is not None else y + 0.28, label,
                ha="center", va="center", fontsize=fs, weight="bold")


def sub(ax, x, y, w, h, caption=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.18",
                                fill=False, ec="#888888", lw=1.0, ls=(0, (1.5, 2))))
    if caption:
        ax.text(x + w / 2, y + 0.2, caption, ha="center", va="center", fontsize=9.5,
                weight="bold", color="#222222")


def card(ax, x, y, w, h, title=None, bullets=(), fill=BLUE, fs=9.5, center=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=EDGE, lw=1.0))
    lines = []
    if title:
        lines.append(title)
    if center:
        ax.text(x + w / 2, y + h / 2, NL.join(lines + list(bullets)), ha="center", va="center",
                fontsize=fs, weight="bold" if title and not bullets else "normal", linespacing=1.35)
        return
    block = (0.34 if title else 0) + 0.29 * len(bullets)
    ty = y + h / 2 + block / 2 + 0.02
    if title:
        ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=fs + 0.5, weight="bold")
        ty -= 0.34
    for b in bullets:
        ax.text(x + 0.18, ty, u"•  " + b, ha="left", va="top", fontsize=fs)
        ty -= 0.29


def thumb(ax, img, x, y, s):
    ax.imshow(img, extent=(x, x + s, y, y + s), zorder=3)
    ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, ec="#444444", lw=0.8, zorder=4))


def arrow(ax, p, q, rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=13, lw=lw, color="#333333",
                                 connectionstyle=f"arc3,rad={rad}", zorder=5))


def poly_arrow(ax, pts, lw=1.4):
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#333333", lw=lw, zorder=5)
    arrow(ax, pts[-2], pts[-1], lw=lw)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- figure 1
def fig_system(t):
    fig, ax = canvas(16, 10.4)

    # (a) data acquisition, top-left
    panel(ax, 0.2, 7.0, 10.6, 3.2, "(a) Data Acquisition")
    sub(ax, 0.45, 7.65, 3.3, 2.35, "Laboratory images")
    thumb(ax, t["pv"], 0.6, 7.95, 1.3)
    card(ax, 2.0, 8.0, 1.6, 1.85, "PlantVillage", ["54,305", "38 classes", "plain"], fs=8.5)
    sub(ax, 3.95, 7.65, 3.3, 2.35, "Field images")
    thumb(ax, t["pd"], 4.1, 7.95, 1.3)
    card(ax, 5.5, 8.0, 1.6, 1.85, "PlantDoc", ["2,580", "28 classes", "cluttered"], fs=8.5)
    sub(ax, 7.45, 7.65, 3.15, 2.35, "Lesion annotations")
    thumb(ax, t["ps"], 7.6, 7.95, 1.3)
    card(ax, 9.0, 8.0, 1.45, 1.85, "PlantSeg", ["11,458", "human", "masks"], fs=8.5)

    # (b) mask generation and data preparation, right column
    panel(ax, 11.2, 0.2, 4.6, 10.0, "(b) Masks and Data Preparation", label_y=0.55)
    sub(ax, 11.4, 7.3, 4.2, 2.7, "Leaf masks")
    thumb(ax, t["mask"], 11.55, 7.75, 1.5)
    card(ax, 13.2, 7.75, 2.25, 2.05, None, ["SAM ViT-B", "centre-point", "prompt", "colour fallback"], fs=8.8)
    sub(ax, 11.4, 4.45, 4.2, 2.6, "Lesion masks")
    thumb(ax, t["ps"], 11.55, 4.9, 1.5)
    card(ax, 13.2, 4.9, 2.25, 1.95, None, ["U-Net", "ResNet-34", "IoU 0.565", "Dice 0.687"], fs=8.8)
    sub(ax, 11.4, 1.1, 4.2, 3.1, "Background swap set")
    thumb(ax, t["cf"], 11.55, 1.95, 1.5)
    card(ax, 13.2, 1.55, 2.25, 2.45, None, ["same leaf,", "new background", "5 conditions", "200 leaves each"], fs=8.8)
    arrow(ax, (13.5, 7.3), (13.5, 7.05))
    arrow(ax, (13.5, 4.45), (13.5, 4.2))

    # (c) training and explainable AI, bottom-left
    panel(ax, 0.2, 0.2, 10.6, 6.5, "(c) Training, Explainable AI and Evaluation", label_y=0.55)
    sub(ax, 0.45, 4.2, 6.2, 2.25, "Deep learning classifier")
    card(ax, 0.65, 4.55, 2.2, 1.7, "ResNet-50", ["ImageNet init", "224 x 224", "AdamW"], fill=GREY, fs=8.8)
    card(ax, 3.05, 4.55, 3.45, 1.7, "Training regimes",
         ["Baseline", "Background removal", "OffLeaf attention loss", "Few-shot field adaptation"], fill=BEIGE, fs=8.8)
    arrow(ax, (2.85, 5.4), (3.05, 5.4))

    sub(ax, 0.45, 2.75, 6.2, 1.25, None)
    card(ax, 1.3, 2.95, 4.5, 0.6, "Disease prediction (top-5 classes)", center=True, fill=GREY, fs=9.5)
    arrow(ax, (3.55, 4.2), (3.55, 3.55))

    card(ax, 0.45, 1.0, 4.7, 1.5, None, [], fill="white")
    thumb(ax, t["cam"], 0.6, 1.1, 1.3)
    ax.text(3.45, 2.2, "Explainable AI", ha="center", fontsize=10, weight="bold")
    ax.text(3.45, 1.7, "Grad-CAM / HiResCAM heatmap" + NL + "off-leaf attention %", ha="center",
            va="center", fontsize=8.8, linespacing=1.3)
    arrow(ax, (2.6, 2.95), (2.6, 2.5))

    card(ax, 5.35, 1.0, 1.3, 1.5, "Web" + NL + "GUI" + NL + "demo", center=True, fill=BEIGE, fs=9.5)
    arrow(ax, (5.15, 1.75), (5.35, 1.75))

    sub(ax, 6.95, 1.0, 3.65, 5.45, "Evaluation library")
    card(ax, 7.15, 4.55, 3.25, 1.6, "Accuracy", ["lab and field", "macro F1", "bootstrap 95% CI"], fs=8.8)
    card(ax, 7.15, 2.85, 3.25, 1.5, "Background reliance", ["corner pixel test", "background swap", "gap decomposition"], fs=8.8)
    card(ax, 7.15, 1.45, 3.25, 1.2, "Attention", ["off-leaf mass"], fs=8.8)
    arrow(ax, (6.5, 5.4), (7.15, 5.4))
    arrow(ax, (6.65, 1.9), (7.15, 1.9))

    # cross-panel arrows
    poly_arrow(ax, [(10.8, 8.8), (11.0, 8.8), (11.4, 8.8)])
    poly_arrow(ax, [(11.4, 5.9), (11.0, 5.9), (11.0, 6.85), (6.2, 6.85), (6.2, 6.25)])
    poly_arrow(ax, [(11.4, 2.6), (10.6, 2.6)])
    arrow(ax, (1.75, 7.65), (1.75, 6.25))

    ax.text(8.0, 10.35, "OffLeaf: Proposed System Architecture", ha="center", fontsize=15, weight="bold")
    save(fig, "fig_arch_system.png")


# ---------------------------------------------------------------- figure 2
def fig_workflow(t):
    fig, ax = canvas(14, 14.6)

    # row 1: datasets
    panel(ax, 0.2, 10.8, 13.6, 3.4, "Datasets", label_y=11.12)
    sub(ax, 0.45, 11.4, 4.2, 2.6, "Laboratory (training)")
    thumb(ax, t["pv"], 0.6, 11.75, 1.6)
    card(ax, 2.35, 11.75, 2.15, 2.05, "PlantVillage", ["54,305 images", "38 classes", "14 crops"], fs=8.8)
    sub(ax, 4.9, 11.4, 4.2, 2.6, "Field (testing, few-shot)")
    thumb(ax, t["pd"], 5.05, 11.75, 1.6)
    card(ax, 6.8, 11.75, 2.15, 2.05, "PlantDoc", ["2,580 images", "28 classes", "real farms"], fs=8.8)
    sub(ax, 9.35, 11.4, 4.2, 2.6, "Lesion annotations")
    thumb(ax, t["ps"], 9.5, 11.75, 1.6)
    card(ax, 11.25, 11.75, 2.15, 2.05, "PlantSeg", ["11,458 images", "human lesion", "masks"], fs=8.8)
    arrow(ax, (7.0, 10.8), (7.0, 10.45))

    # row 2: preparation
    panel(ax, 0.2, 7.5, 13.6, 2.9, "Data Preparation, Mask Generation and Data Mapping", label_y=7.82)
    sub(ax, 0.45, 8.1, 4.2, 2.1, "Data splits")
    card(ax, 0.6, 8.45, 3.9, 1.6, None, ["80 / 10 / 10 stratified", "PlantVillage to PlantDoc", "class mapping"], fs=9)
    sub(ax, 4.9, 8.1, 4.2, 2.1, "Mask generation")
    card(ax, 5.05, 8.45, 3.9, 1.6, None, ["SAM leaf masks", "U-Net lesion masks", "(trained on PlantSeg)"], fs=9)
    sub(ax, 9.35, 8.1, 4.2, 2.1, "Augmentation")
    card(ax, 9.5, 8.45, 3.9, 1.6, None, ["crops, flips, rotations", "masks move with image", "background swap set"], fs=9)
    arrow(ax, (4.5, 9.25), (5.05, 9.25))
    arrow(ax, (8.95, 9.25), (9.5, 9.25))
    arrow(ax, (7.0, 7.5), (7.0, 7.15))

    # row 3: training regimes
    panel(ax, 0.2, 4.95, 13.6, 2.15, "Training regimes (ResNet-50)", label_y=5.27)
    names = ["Baseline", "Background" + NL + "removal", "OffLeaf" + NL + "attention loss", "Few-shot field" + NL + "adaptation"]
    fills = [GREY, BLUE, BEIGE, BEIGE]
    for i, (n, f) in enumerate(zip(names, fills)):
        card(ax, 0.55 + i * 3.35, 5.6, 3.0, 1.2, n, center=True, fill=f, fs=10)

    # branches
    poly_arrow(ax, [(7.0, 4.95), (7.0, 4.7), (3.45, 4.7), (3.45, 4.35)])
    poly_arrow(ax, [(7.0, 4.7), (10.55, 4.7), (10.55, 4.35)])

    panel(ax, 0.2, 0.2, 6.5, 4.15, None)
    sub(ax, 0.45, 2.3, 6.0, 1.85, "Disease classification")
    card(ax, 0.6, 2.65, 5.7, 1.3, None, ["38 disease and healthy classes", "top-5 prediction with confidence"], fill=BEIGE, fs=9)
    arrow(ax, (3.45, 2.3), (3.45, 2.05))
    sub(ax, 0.45, 0.4, 6.0, 1.65, "Classification performance")
    card(ax, 0.6, 0.75, 5.7, 1.1, None, ["accuracy and macro F1, lab and field", "bootstrap 95% confidence intervals"], fs=9)

    panel(ax, 7.3, 0.2, 6.5, 4.15, None)
    sub(ax, 7.55, 2.3, 6.0, 1.85, "Background reliance analysis")
    card(ax, 7.7, 2.65, 5.7, 1.3, None, ["corner pixel test", "background swap benchmark"], fill=BEIGE, fs=9)
    thumb(ax, t["cf"], 12.25, 2.75, 1.05)
    arrow(ax, (10.55, 2.3), (10.55, 2.05))
    sub(ax, 7.55, 0.4, 6.0, 1.65, "Explainability evaluation")
    card(ax, 7.7, 0.75, 5.7, 1.1, None, ["Grad-CAM off-leaf attention %", "background share of lab to field gap"], fs=9)
    thumb(ax, t["cam"], 12.35, 0.8, 1.0)

    ax.text(7.0, 14.5, "OffLeaf: Objective-wise Workflow", ha="center", fontsize=15, weight="bold")
    save(fig, "fig_arch_workflow.png")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, facecolor="white")
    if REPORT_FIGS.exists():
        shutil.copy(OUT / name, REPORT_FIGS / name)
    plt.close(fig)
    print("wrote", name)


if __name__ == "__main__":
    t = thumbs()
    fig_system(t)
    fig_workflow(t)
