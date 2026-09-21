"""Gradio demo for an OffLeaf checkpoint.

Two tabs:

**Classify** - upload a leaf, get the top-3 classes with confidence and a
Grad-CAM overlay showing where the model looked. Warns when top confidence is
below 0.6.

**Background swap** - the tab that carries the argument. Pick an image from the
counterfactual benchmark and see the model's prediction on the *same leaf*
against five different backgrounds. When the answer changes and the leaf did
not, the background was doing the work. No heatmap needed, and nothing to take
on trust.

Usage::

    python release/demo.py --checkpoint runs/E0_resnet50/0/checkpoint.pt
    python release/demo.py --checkpoint ... --share      # public link for a talk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import get_device  # noqa: E402
from data.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from explain.methods import explain  # noqa: E402
from models.build import build_model  # noqa: E402

CELLS = ("lab_plain", "paste_control", "lab_field", "field_plain", "field_field")
CELL_LABEL = {
    "lab_plain": "Lab leaf, lab background",
    "paste_control": "Lab leaf, its OWN background (control)",
    "lab_field": "Lab leaf, FIELD background",
    "field_plain": "Field leaf, lab background",
    "field_field": "Field leaf, field background",
}
LOW_CONFIDENCE = 0.6


def load_class_names(repo: Path) -> list[str]:
    p = repo / "data" / "splits" / "plantvillage_classes.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p).sort_values("label")
    return [str(c) for c in df.class_name]


def prettify(name: str) -> str:
    """`Tomato___Late_blight` -> `Tomato - Late blight`."""
    return name.replace("___", " - ").replace("_", " ").strip()


def preprocess(rgb: np.ndarray, size: int = 224) -> torch.Tensor:
    img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = img.astype(np.float32) / 255.0
    x = (x - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


def overlay_cam(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Jet heatmap blended over the image."""
    h, w = rgb.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


class Demo:
    def __init__(self, checkpoint: Path, repo: Path):
        self.repo = repo
        self.device = get_device()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = build_model(
            ck["model_name"], num_classes=ck["num_classes"], pretrained=False
        )
        self.model.load_state_dict(ck["model_state"])
        self.model.to(self.device).eval()
        self.meta = {
            "exp_id": ck.get("exp_id"),
            "regime": ck.get("config", {}).get("regime"),
            "lam": ck.get("config", {}).get("lam"),
            "val_acc": ck.get("val_acc"),
            "num_classes": ck["num_classes"],
        }
        names = load_class_names(repo)
        self.class_names = names if len(names) == ck["num_classes"] else [
            f"class_{i}" for i in range(ck["num_classes"])
        ]

    @torch.no_grad()
    def _probs(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb).to(self.device)
        logits, _ = self.model(x)
        return logits.float().softmax(dim=1)[0].cpu().numpy()

    def classify(self, image):
        """Top-3 + Grad-CAM overlay + an honesty note."""
        if image is None:
            return None, {}, "Upload a leaf photo to begin."
        rgb = np.asarray(image.convert("RGB"))
        probs = self._probs(rgb)
        order = np.argsort(-probs)[:3]
        top3 = {prettify(self.class_names[i]): float(probs[i]) for i in order}

        x = preprocess(rgb).to(self.device)
        cam = explain(self.model, x, method="gradcam")[0].cpu().numpy()
        overlay = overlay_cam(rgb, cam)

        top_conf = float(probs[order[0]])
        if top_conf < LOW_CONFIDENCE:
            note = (
                f"**Low confidence ({top_conf:.0%}).** Treat this prediction as unreliable. "
                "The model has no way to abstain, so it always returns a class."
            )
        else:
            note = (
                f"Top confidence {top_conf:.0%}. **High confidence is not evidence of "
                "correctness here** - this model scores ~99% in the lab and ~20% on real "
                "field photos while staying confident. Check the heatmap: if the heat sits "
                "off the leaf, the prediction is being driven by background."
            )
        return overlay, top3, note

    # ---------------- background swap ----------------

    def counterfactual_ids(self) -> list[str]:
        idx = self.repo / "data" / "counterfactual" / "index.csv"
        if not idx.exists():
            return []
        df = pd.read_csv(idx)
        lab = df[(df.cell == "lab_plain") & (df.source_dataset == "plantvillage")]
        return sorted(lab.image_id.unique())[:200]

    def swap(self, image_id: str):
        """Predictions for one leaf across every background cell."""
        root = self.repo / "data" / "counterfactual"
        if not image_id:
            return [], "Pick an image."

        gallery, rows, preds = [], [], []
        for cell in CELLS:
            p = root / cell / f"{image_id}.jpg"
            if not p.exists():
                continue
            bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            probs = self._probs(rgb)
            i = int(np.argmax(probs))
            name = prettify(self.class_names[i])
            preds.append(i)
            gallery.append((rgb, f"{CELL_LABEL[cell]}\n-> {name} ({probs[i]:.0%})"))
            rows.append(f"| {CELL_LABEL[cell]} | {name} | {probs[i]:.0%} |")

        if not gallery:
            return [], f"No composites found for `{image_id}`."

        n_unique = len(set(preds))
        verdict = (
            f"### The leaf is pixel-identical in every image above.\n\n"
            f"The model gave **{n_unique} different answer(s)** across "
            f"{len(preds)} backgrounds.\n\n"
        )
        verdict += (
            "**The prediction changed when only the background changed.** "
            "Nothing about the leaf differs - so the background was doing the work.\n\n"
            if n_unique > 1
            else "The prediction held across every background for this leaf.\n\n"
        )
        table = "| background | prediction | confidence |\n|---|---|---|\n" + "\n".join(rows)
        return gallery, verdict + table


def build_ui(demo: Demo):
    import gradio as gr

    m = demo.meta
    header = (
        f"# OffLeaf\n"
        f"### Does a plant-disease classifier look at the disease, or at the background?\n\n"
        f"Model `{m['exp_id']}` · regime `{m['regime']}`"
        + (f" · lambda `{m['lam']}`" if m.get("lam") else "")
        + f" · {m['num_classes']} classes · lab val **{m['val_acc']:.2%}**\n\n"
        "> **Research baseline, not deployment-ready.** Zero-shot field accuracy is "
        "around 20%. Do not use this to diagnose a real plant."
    )

    with gr.Blocks(title="OffLeaf", theme=gr.themes.Soft()) as ui:
        gr.Markdown(header)

        with gr.Tab("Classify a leaf"):
            with gr.Row():
                with gr.Column():
                    inp = gr.Image(type="pil", label="Leaf photo", height=340)
                    btn = gr.Button("Classify", variant="primary")
                with gr.Column():
                    out_img = gr.Image(label="Where the model looked (Grad-CAM)", height=340)
                    out_lbl = gr.Label(num_top_classes=3, label="Top 3")
            out_note = gr.Markdown()
            btn.click(demo.classify, inputs=inp, outputs=[out_img, out_lbl, out_note])
            inp.change(demo.classify, inputs=inp, outputs=[out_img, out_lbl, out_note])

        with gr.Tab("Background swap"):
            ids = demo.counterfactual_ids()
            if not ids:
                gr.Markdown(
                    "Counterfactual set not built. Run:\n\n"
                    "```\npython counterfactual/build.py "
                    "--config configs/counterfactual_tomato.yaml\n```"
                )
            else:
                gr.Markdown(
                    "**The same leaf, cut out and placed on five different backgrounds.** "
                    "If the model is reading the disease, the answer should not move."
                )
                pick = gr.Dropdown(ids, value=ids[0], label="Leaf")
                run = gr.Button("Show all backgrounds", variant="primary")
                gal = gr.Gallery(label="Same leaf, different backgrounds", columns=5, height=260)
                verdict = gr.Markdown()
                run.click(demo.swap, inputs=pick, outputs=[gal, verdict])
                pick.change(demo.swap, inputs=pick, outputs=[gal, verdict])
    return ui


def main() -> None:
    ap = argparse.ArgumentParser(description="OffLeaf Gradio demo.")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--share", action="store_true", help="public link (useful for a talk)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    ck = Path(args.checkpoint)
    if not ck.is_absolute():
        ck = repo / ck
    if not ck.exists():
        raise SystemExit(f"Checkpoint not found: {ck}")

    demo = Demo(ck, repo)
    print(f"loaded {demo.meta}", flush=True)
    build_ui(demo).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
