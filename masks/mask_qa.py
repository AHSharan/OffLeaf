"""Visual QA sheet for leaf masks.

Writes a self-contained HTML page showing original-vs-overlay pairs, sampled
across each generation method (``sam`` / ``hsv_fallback`` / ``unreliable``) so
the failure modes are visible rather than hidden in an aggregate percentage.

Look for:

* mask following the leaf outline, not the whole frame
* the ``unreliable`` rows - are they genuinely bad, or is a macro shot being
  flagged because the leaf legitimately fills the image?
* anything that is not a leaf photo at all (PlantDoc contains some)

Usage::

    python masks/mask_qa.py --dataset plantvillage
    python masks/mask_qa.py --dataset plantdoc --per-method 8
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METHODS = ("sam", "hsv_fallback", "unreliable")
THUMB_H = 230


def read_log(path: Path) -> list[dict]:
    """Read the mask log, tolerating the pre-``reliable`` 4-field schema."""
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.reader(fh)
        next(r, None)
        for x in r:
            if len(x) < 4:
                continue
            rows.append(
                {
                    "rel": x[0],
                    "method": x[1],
                    "score": x[2],
                    "coverage": float(x[3]),
                    "reliable": x[4] if len(x) >= 5 else "",
                }
            )
    return rows


def overlay(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Original beside a green-tinted, outlined mask overlay."""
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    b = (mask > 127).astype(np.uint8)
    ov = img.copy()
    green = np.zeros_like(ov)
    green[..., 1] = 255
    ov[b == 1] = (0.5 * green[b == 1] + 0.5 * ov[b == 1]).astype(np.uint8)
    cnt, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, (0, 255, 255), 2)
    pair = np.concatenate([img, ov], axis=1)
    s = THUMB_H / pair.shape[0]
    return cv2.resize(pair, (int(pair.shape[1] * s), THUMB_H), interpolation=cv2.INTER_AREA)


HEAD = """<!doctype html>
<meta charset="utf-8"><title>OffLeaf - leaf mask QA</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px;
      background:#141714;color:#e8ebe6}
 h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:26px 0 10px;color:#9ec37d}
 .sub{color:#9aa096;font-size:13px;line-height:1.6;margin-bottom:18px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}
 .c{background:#1c201b;border:1px solid #2d332c;border-radius:8px;padding:8px}
 .c img{width:100%;border-radius:5px;display:block}
 .m{font-size:11px;line-height:1.6;margin-top:6px;color:#9aa096;word-break:break-all}
 .bad{color:#f28b82;font-weight:600}
 code{background:#2d332c;padding:1px 5px;border-radius:4px}
</style>
<h1>Leaf mask QA &mdash; {dataset}</h1>
<div class="sub">
Left = original, right = mask (green fill, yellow outline).<br>
Check that the mask follows the <em>leaf</em> and not the whole frame. Coverage above
~0.85 usually means the mask has grabbed the background too &mdash; though on a macro
shot where the leaf genuinely fills the frame, high coverage is correct and the
<code>unreliable</code> flag is a false positive.
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Visual QA sheet for leaf masks.")
    ap.add_argument("--dataset", default="plantvillage", choices=["plantvillage", "plantdoc"])
    ap.add_argument("--per-method", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    mask_root = repo / "data" / "masks" / "leaf" / args.dataset
    raw_root = repo / "data" / "raw" / args.dataset
    log = mask_root / "_leaf_mask_log.csv"
    if not log.exists():
        raise SystemExit(f"{log} not found. Run masks/leaf_masks.py first.")

    rows = read_log(log)
    rng = random.Random(args.seed)
    print(f"{len(rows)} masks logged for {args.dataset}", flush=True)

    body = []
    for method in METHODS:
        sel = [r for r in rows if r["method"] == method]
        if not sel:
            continue
        pct = 100 * len(sel) / max(len(rows), 1)
        body.append(f"<h2>{html.escape(method)} &mdash; {len(sel)} masks ({pct:.1f}%)</h2>")
        body.append('<div class="grid">')
        for r in rng.sample(sel, min(args.per_method, len(sel))):
            img = cv2.imread(str(raw_root / r["rel"]), cv2.IMREAD_COLOR)
            msk = cv2.imread(str(mask_root / f"{r['rel']}.png"), cv2.IMREAD_GRAYSCALE)
            if img is None or msk is None:
                continue
            ok, buf = cv2.imencode(".jpg", overlay(img, msk), [cv2.IMWRITE_JPEG_QUALITY, 72])
            if not ok:
                continue
            uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
            flag = "" if r["reliable"] != "0" else '<span class="bad">UNRELIABLE</span><br>'
            body.append(
                f'<div class="c"><img src="{uri}" alt="">'
                f'<div class="m">{flag}coverage <b>{r["coverage"]:.3f}</b> &middot; '
                f'SAM score {html.escape(r["score"])}<br>{html.escape(r["rel"])}</div></div>'
            )
        body.append("</div>")

    out = Path(args.out) if args.out else mask_root / "mask_qa.html"
    out.write_text(HEAD.replace("{dataset}", args.dataset) + "\n".join(body), encoding="utf-8")
    print(f"\nwrote {out}\nOpen it in a browser.")


if __name__ == "__main__":
    main()
