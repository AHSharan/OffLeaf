"""Image figures for the report: dataset samples, masks, background swap,
Grad-CAM examples and the demo screenshot. Writes to results/figures/ and
report/Figures/.

    python scripts/make_report_images.py
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from data.dataset import IMAGENET_MEAN, IMAGENET_STD, mask_path_for  # noqa: E402
from explain.methods import explain  # noqa: E402
from models.build import build_model  # noqa: E402

OUT = REPO / "results" / "figures"
REPORT_FIGS = REPO / "report" / "Figures"
plt.rcParams.update({"font.family": "serif",
                     "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 10, "savefig.dpi": 250, "savefig.bbox": "tight"})
random.seed(3)


def save(fig, name):
    fig.savefig(OUT / name)
    shutil.copy(OUT / name, REPORT_FIGS / name)
    plt.close(fig)
    print("wrote", name)


def rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def pretty(name):
    return name.replace("___", ": ").replace("_", " ")


def pick(root: Path, classes, n=1):
    out = []
    for c in classes:
        files = sorted((root / c).glob("*"))
        out.append(random.choice(files))
    return out


# ---------------------------------------------------------------- samples
def fig_samples():
    pv = REPO / "data/raw/plantvillage/color"
    pd_ = REPO / "data/raw/plantdoc/train"
    pv_cls = ["Tomato___Early_blight", "Tomato___Late_blight", "Apple___Apple_scab",
              "Corn_(maize)___Common_rust_"]
    pd_cls = ["Tomato Early blight leaf", "Tomato leaf late blight", "Apple Scab Leaf",
              "Corn rust leaf"]
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2))
    for j, (p, c) in enumerate(zip(pick(pv, pv_cls), pv_cls)):
        axes[0, j].imshow(cv2.resize(rgb(p), (256, 256))); axes[0, j].set_title(pretty(c), fontsize=9)
    for j, (p, c) in enumerate(zip(pick(pd_, pd_cls), pd_cls)):
        axes[1, j].imshow(cv2.resize(rgb(p), (256, 256))); axes[1, j].set_title(c, fontsize=9)
    for a in axes.ravel():
        a.axis("off")
    fig.text(0.005, 0.73, "PlantVillage\n(lab)", fontsize=11, weight="bold", va="center")
    fig.text(0.005, 0.27, "PlantDoc\n(field)", fontsize=11, weight="bold", va="center")
    fig.subplots_adjust(left=0.1, wspace=0.05, hspace=0.18)
    save(fig, "fig_samples.png")


# ---------------------------------------------------------------- masks
def overlay(img, m, color):
    ov = img.copy()
    sel = m > 127
    ov[sel] = (0.5 * np.array(color) + 0.5 * ov[sel]).astype(np.uint8)
    cnt, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, (255, 230, 0), 2)
    return ov


def fig_masks():
    import pandas as pd
    root = REPO / "data/raw/plantvillage"
    conf = pd.read_csv(REPO / "data/masks/lesion_pseudo/pseudo_confidence.csv")
    cls = ["Tomato___Early_blight", "Tomato___Bacterial_spot"]
    fig, axes = plt.subplots(2, 3, figsize=(8, 5.4))
    for i, c in enumerate(cls):
        sub = conf[(conf.class_name == c) & (conf.fg_fraction / conf.leaf_coverage).between(0.08, 0.3)
                   & (conf.mean_fg_prob > 0.85)]
        p = root / sub.sample(1, random_state=4 + i).iloc[0]["path"]
        img = rgb(p)
        leaf = cv2.imread(str(mask_path_for(p, REPO / "data/masks/leaf")), 0)
        les = cv2.imread(str(mask_path_for(p, REPO / "data/masks/lesion_pseudo")), 0)
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(overlay(img, leaf, (60, 200, 60)))
        axes[i, 2].imshow(overlay(img, les, (220, 40, 40)))
    for j, t in enumerate(["Input image", "Leaf mask (SAM)", "Lesion mask (U-Net)"]):
        axes[0, j].set_title(t)
    for a in axes.ravel():
        a.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    save(fig, "fig_masks.png")


# ---------------------------------------------------------------- counterfactual
def fig_cf_examples():
    cf = REPO / "data/counterfactual"
    ids = sorted(p.stem for p in (cf / "lab_plain").glob("*.jpg"))
    cells = [("paste_control", "Own background\n(control)"), ("lab_plain", "Lab background"),
             ("lab_field", "Field background")]
    chosen = [ids[5], ids[60], ids[140]]
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.4))
    for i, iid in enumerate(chosen):
        for j, (c, t) in enumerate(cells):
            axes[i, j].imshow(rgb(cf / c / f"{iid}.jpg"))
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(t)
    fig.subplots_adjust(wspace=0.04, hspace=0.04)
    save(fig, "fig_cf_examples.png")


# ---------------------------------------------------------------- grad-cam
def load_model():
    ck = torch.load(REPO / "runs/E0_resnet50/0/checkpoint.pt", map_location="cpu", weights_only=False)
    m = build_model(ck["model_name"], num_classes=ck["num_classes"], pretrained=False)
    m.load_state_dict(ck["model_state"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return m.to(dev).eval(), dev


def to_tensor(img):
    x = cv2.resize(img, (224, 224)).astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, np.float32)) / np.array(IMAGENET_STD, np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def heat(img, cam):
    h, w = img.shape[:2]
    c = cv2.resize(cam, (w, h))
    hm = cv2.cvtColor(cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (0.45 * hm + 0.55 * img).astype(np.uint8)


def offleaf(cam, leaf):
    lf = cv2.resize((leaf > 127).astype(np.float32), (cam.shape[1], cam.shape[0]))
    return 1 - float((cam * lf).sum() / max(cam.sum(), 1e-8))


def fig_gradcam():
    model, dev = load_model()
    pv = REPO / "data/raw/plantvillage/color"
    pd_ = REPO / "data/raw/plantdoc/test"
    lab = random.choice(sorted((pv / "Tomato___Late_blight").glob("*")))

    # the field example with the most attention off the leaf, among tomato images
    import csv as _csv
    ok = {}
    with open(REPO / "data/masks/leaf/plantdoc/_leaf_mask_log.csv", newline="", encoding="utf-8") as fh:
        rd = _csv.reader(fh); next(rd)
        for r in rd:
            if len(r) >= 5 and r[4] == "1" and 0.15 <= float(r[3]) <= 0.6 and r[1] == "sam":
                ok[r[0]] = float(r[3])
    best = None
    for p in sorted((REPO / "data/raw/plantdoc").glob("*/Tomato*/*")):
        rel = p.relative_to(REPO / "data/raw/plantdoc").as_posix()
        if rel not in ok:
            continue
        lm = mask_path_for(p, REPO / "data/masks/leaf")
        if not lm.exists():
            continue
        img = rgb(p)
        cam = explain(model, to_tensor(img).to(dev), method="gradcam")[0].cpu().numpy()
        o = offleaf(cam, cv2.imread(str(lm), 0))
        if best is None or o > best[0]:
            best = (o, p, img, cam)

    img_lab = rgb(lab)
    cam_lab = explain(model, to_tensor(img_lab).to(dev), method="gradcam")[0].cpu().numpy()
    o_lab = offleaf(cam_lab, cv2.imread(str(mask_path_for(lab, REPO / "data/masks/leaf")), 0))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.2))
    axes[0, 0].imshow(img_lab); axes[0, 0].set_title("Lab image")
    axes[0, 1].imshow(heat(img_lab, cam_lab)); axes[0, 1].set_title("Grad-CAM")
    axes[1, 0].imshow(best[2]); axes[1, 0].set_title("Field image")
    axes[1, 1].imshow(heat(best[2], best[3])); axes[1, 1].set_title("Grad-CAM")
    for a in axes.ravel():
        a.axis("off")
    fig.subplots_adjust(wspace=0.04, hspace=0.12)
    save(fig, "fig_gradcam.png")


def fig_demo():
    src = REPORT_FIGS / "demo_raw.webp"
    if src.exists():
        img = cv2.imread(str(src))
        cv2.imwrite(str(OUT / "fig_demo.png"), img)
        shutil.copy(OUT / "fig_demo.png", REPORT_FIGS / "fig_demo.png")
        print("wrote fig_demo.png")


if __name__ == "__main__":
    fig_samples(); fig_masks(); fig_cf_examples(); fig_gradcam(); fig_demo()
