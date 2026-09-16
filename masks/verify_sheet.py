"""Build a human verification contact sheet for pseudo lesion masks.

Samples N pseudo-masks (default 80, stratified by class), renders each as
original-vs-overlay, and writes a self-contained HTML sheet plus a ``verify.csv``
for a person to fill in with accept / fix / reject.

This is the gate that turns training-only pseudo masks into metric-grade human
masks. Nothing reaches ``lesion_human/`` without passing through here and then
``import_verified.py``.

Images are embedded as base64 data URIs so the sheet is a single file that can
be emailed or opened on another machine without the dataset present.

Usage::

    python masks/verify_sheet.py --config configs/verify_tomato.yaml
    python masks/verify_sheet.py --config configs/verify_tomato.yaml --n 120
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, set_seed  # noqa: E402

THUMB_H = 260


def _overlay(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Original beside a red-tinted lesion overlay with its outline drawn."""
    over = image_rgb.copy()
    red = np.zeros_like(over)
    red[..., 0] = 255
    sel = mask > 0
    over[sel] = (0.55 * red[sel] + 0.45 * over[sel]).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(over, contours, -1, (255, 255, 0), 2)

    pair = np.concatenate([image_rgb, over], axis=1)
    scale = THUMB_H / pair.shape[0]
    return cv2.resize(pair, (int(pair.shape[1] * scale), THUMB_H), interpolation=cv2.INTER_AREA)


def _data_uri(rgb: np.ndarray, quality: int = 72) -> str:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed while building the contact sheet")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n rows spread evenly across ``class_name``."""
    classes = sorted(df.class_name.unique())
    per = max(1, n // max(len(classes), 1))
    parts = []
    for c in classes:
        sub = df[df.class_name == c]
        parts.append(sub.sample(min(per, len(sub)), random_state=seed))
    out = pd.concat(parts)
    if len(out) < n:  # top up from whatever is left, to hit n
        rest = df.drop(out.index)
        if len(rest):
            out = pd.concat([out, rest.sample(min(n - len(out), len(rest)), random_state=seed)])
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


HTML_HEAD = """<!doctype html>
<meta charset="utf-8">
<title>OffLeaf - pseudo mask verification</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px;
      background:#14161a;color:#e8eaed;}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:#9aa0a6;font-size:13px;margin-bottom:20px;line-height:1.6}
 .card{background:#1e2126;border:1px solid #2d3138;border-radius:10px;padding:12px;
       margin-bottom:14px;display:flex;gap:16px;align-items:flex-start}
 .card img{border-radius:6px;display:block;max-width:100%}
 .meta{font-size:12px;line-height:1.7;min-width:230px}
 .id{font-weight:600;color:#8ab4f8;word-break:break-all;font-size:11px}
 .k{color:#9aa0a6}
 code{background:#2d3138;padding:1px 5px;border-radius:4px;font-size:11px}
 .warn{color:#fdd663}
</style>
<h1>Pseudo lesion masks - verification sheet</h1>
<div class="sub">
Left = original, right = predicted lesion (red fill, yellow outline).<br>
Fill <code>decision</code> in <code>verify.csv</code> with
<code>accept</code>, <code>fix</code> or <code>reject</code>, then run
<code>masks/import_verified.py</code>.<br>
<span class="warn">These are model output, not ground truth. No reported metric may use them until verified.</span>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the pseudo-mask verification sheet.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=None, help="how many masks to sample")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    n = args.n if args.n is not None else int(cfg.get("n_samples", 80))
    pseudo_root = repo / cfg.get("pseudo_root", "data/masks/lesion_pseudo")
    image_root = repo / cfg.get("image_root", "data/raw/plantvillage")
    dataset = cfg.get("dataset_name", "plantvillage")
    out_dir = repo / cfg.get("out_dir", "data/masks/verify")
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_path = pseudo_root / "pseudo_confidence.csv"
    if not conf_path.exists():
        raise FileNotFoundError(
            f"{conf_path} not found. Run masks/pseudo_label.py first."
        )
    df = pd.read_csv(conf_path)
    if df.empty:
        raise ValueError(f"{conf_path} is empty - no pseudo masks to verify.")

    sample = stratified_sample(df, n, seed)
    print(f"sampling {len(sample)} of {len(df)} pseudo masks across "
          f"{sample.class_name.nunique()} classes", flush=True)

    cards, rows = [], []
    for r in sample.to_dict("records"):
        rel = r["path"]
        img_path = image_root / rel
        mask_path = pseudo_root / dataset / f"{rel}.png"

        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"Could not read image referenced by the confidence csv: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise OSError(f"Pseudo mask missing: {mask_path}")
        mask = (m > 127).astype(np.uint8)
        if mask.shape[:2] != rgb.shape[:2]:
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        uri = _data_uri(_overlay(rgb, mask))
        cards.append(
            f'<div class="card"><img src="{uri}" alt="">'
            f'<div class="meta"><div class="id">{html.escape(rel)}</div>'
            f'<div><span class="k">class</span> {html.escape(str(r["class_name"]))}</div>'
            f'<div><span class="k">mean fg prob</span> {r["mean_fg_prob"]}</div>'
            f'<div><span class="k">fg fraction</span> {r["fg_fraction"]}</div>'
            f'<div><span class="k">leaf coverage</span> {r["leaf_coverage"]}</div>'
            f'<div><span class="k">clipped away</span> {r["clipped_away"]}</div>'
            f"</div></div>"
        )
        rows.append({"id": rel, "class_name": r["class_name"], "decision": "", "note": ""})

    sheet = out_dir / "verify_sheet.html"
    sheet.write_text(HTML_HEAD + "\n".join(cards), encoding="utf-8")

    vcsv = out_dir / "verify.csv"
    with open(vcsv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "class_name", "decision", "note"])
        w.writeheader()
        w.writerows(rows)

    print(f"\ncontact sheet: {sheet}")
    print(f"decisions csv: {vcsv}")
    print("\nOpen the sheet, fill 'decision' in verify.csv with accept|fix|reject, then run:")
    print("  python masks/import_verified.py --config <config>")


if __name__ == "__main__":
    main()
