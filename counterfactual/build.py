"""Build the 2x2 background-swap counterfactual benchmark.

The core measurement instrument. The same leaf is composited onto backgrounds it
did not come from; if a classifier's prediction changes, it was relying on the
background rather than the leaf.

Cells (spec section 4):

===============  ====================================================
``lab_plain``    lab leaf   on an inpainted PlantVillage background
``lab_field``    lab leaf   on an inpainted field background
``field_plain``  field leaf on an inpainted PlantVillage background
``field_field``  field leaf on an inpainted field background
``paste_control` leaf back onto its *own* inpainted background
===============  ====================================================

``paste_control`` is the one that makes the rest interpretable. It applies every
compositing artefact - cutout, feather, rescale, reposition - but changes no
background. Prediction changes there are caused by the compositing pipeline, so
it is the floor against which the other cells must be read. Without it a
"background effect" could just be feathering.

Only leaf masks flagged ``reliable=1`` by ``masks/leaf_masks.py`` are used. A
cutout from a mask covering 90% of a field photo is a picture of a field, not a
leaf, and would silently weaken every number computed here.

Usage::

    python counterfactual/build.py --config configs/counterfactual_tomato.yaml
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
from data.dataset import mask_path_for, split_raw_path  # noqa: E402

CELLS = ("lab_plain", "lab_field", "field_plain", "field_field", "paste_control")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# --------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise OSError(f"Could not read mask: {path}")
    if m.shape[:2] != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def reliable_ids(log_path: Path) -> set[str] | None:
    """Relative paths whose leaf mask is usable, or None if the log predates the flag."""
    if not log_path.exists():
        return None
    df = pd.read_csv(log_path)
    if "reliable" not in df.columns:
        print(
            f"WARNING: {log_path} has no 'reliable' column - it predates the mask reliability "
            "fix. Regenerate leaf masks, or cutouts may be built on whole-scene masks.",
            flush=True,
        )
        return None
    return set(df[df.reliable == 1].relative_path)


# --------------------------------------------------------------------------
# background bank
# --------------------------------------------------------------------------


def inpaint_background(image: np.ndarray, leaf: np.ndarray, radius: int = 5) -> np.ndarray:
    """Remove the leaf and fill the hole, giving a leaf-free background plate.

    The mask is dilated before inpainting so the leaf's own edge pixels do not
    survive and bleed a green halo into the filled region.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hole = cv2.dilate(leaf, kernel, iterations=2)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filled = cv2.inpaint(bgr, hole, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(filled, cv2.COLOR_BGR2RGB)


def texture_background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """A solid colour or simple synthetic texture."""
    kind = rng.integers(0, 3)
    base = rng.integers(40, 210, size=3).astype(np.float32)

    if kind == 0:  # flat colour with light noise
        img = np.clip(base + rng.normal(0, 4, (h, w, 3)), 0, 255)
    elif kind == 1:  # linear gradient between two colours
        other = np.clip(base + rng.integers(-60, 60, size=3), 0, 255).astype(np.float32)
        t = np.linspace(0, 1, w, dtype=np.float32)[None, :, None]
        img = base[None, None, :] * (1 - t) + other[None, None, :] * t
        img = np.repeat(img, h, axis=0) + rng.normal(0, 3, (h, w, 3))
    else:  # coarse blobby texture, upsampled low-res noise
        small = rng.normal(0, 1, (max(h // 32, 2), max(w // 32, 2), 3)).astype(np.float32)
        img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) * 28 + base

    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# compositing
# --------------------------------------------------------------------------


def composite(
    leaf_img: np.ndarray,
    leaf_mask: np.ndarray,
    background: np.ndarray,
    rng: np.random.Generator,
    feather_px: int = 4,
    jitter: float = 0.15,
    lesion_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Paste a feathered leaf cutout onto ``background``.

    Returns ``(image, moved_leaf_mask, moved_lesion_mask)`` - the masks are
    transformed identically to the leaf, so relevance mass stays computable on
    the composite.

    Raises:
        ValueError: If the leaf mask is empty (nothing to cut out).
    """
    ys, xs = np.where(leaf_mask > 0)
    if ys.size == 0:
        raise ValueError("Empty leaf mask - nothing to composite")

    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = leaf_img[y0:y1, x0:x1]
    cmask = leaf_mask[y0:y1, x0:x1]
    clesion = lesion_mask[y0:y1, x0:x1] if lesion_mask is not None else None

    H, W = background.shape[:2]
    ch, cw = crop.shape[:2]

    # Fit the cutout inside the background, then jitter scale by +/- jitter.
    fit = min(W / cw, H / ch, 1.0) * 0.8
    scale = float(fit * (1.0 + rng.uniform(-jitter, jitter)))
    nw, nh = max(8, int(cw * scale)), max(8, int(ch * scale))

    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
    cmask = cv2.resize(cmask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if clesion is not None:
        clesion = cv2.resize(clesion, (nw, nh), interpolation=cv2.INTER_NEAREST)

    # Random position, biased to the centre by the +/- jitter range.
    max_x, max_y = max(W - nw, 0), max(H - nh, 0)
    cx = int(np.clip((W - nw) / 2 + rng.uniform(-jitter, jitter) * W, 0, max_x))
    cy = int(np.clip((H - nh) / 2 + rng.uniform(-jitter, jitter) * H, 0, max_y))

    # Feather the alpha so the cutout edge is not a hard, learnable seam.
    k = max(3, int(feather_px) | 1)
    alpha = cv2.GaussianBlur(cmask.astype(np.float32), (k, k), 0)[..., None]

    out = background.copy()
    region = out[cy : cy + nh, cx : cx + nw].astype(np.float32)
    out[cy : cy + nh, cx : cx + nw] = (
        alpha * crop.astype(np.float32) + (1 - alpha) * region
    ).astype(np.uint8)

    moved_leaf = np.zeros((H, W), dtype=np.uint8)
    moved_leaf[cy : cy + nh, cx : cx + nw] = cmask
    moved_lesion = None
    if clesion is not None:
        moved_lesion = np.zeros((H, W), dtype=np.uint8)
        moved_lesion[cy : cy + nh, cx : cx + nw] = clesion

    return out, moved_leaf, moved_lesion


# --------------------------------------------------------------------------
# contact sheet
# --------------------------------------------------------------------------

SHEET_HEAD = """<!doctype html>
<meta charset="utf-8"><title>OffLeaf - counterfactual QA</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:0;padding:24px;
      background:#14161a;color:#e8eaed}
 h1{font-size:20px;margin:0 0 4px}
 .sub{color:#9aa0a6;font-size:13px;margin-bottom:20px;line-height:1.6}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
 .c{background:#1e2126;border:1px solid #2d3138;border-radius:8px;padding:8px}
 .c img{width:100%;border-radius:5px;display:block}
 .m{font-size:11px;line-height:1.6;margin-top:6px;color:#9aa0a6}
 .cell{color:#8ab4f8;font-weight:600}
</style>
<h1>Counterfactual composites - visual QA</h1>
<div class="sub">Check for: hard seams at the leaf edge, green halos from inpainting,
leaves cropped by the frame, and backgrounds that still contain the original leaf.</div>
<div class="grid">
"""


def write_contact_sheet(rows: list[dict], out_dir: Path, path: Path, n: int, rng) -> None:
    """Render n random composites for visual inspection."""
    sample = rng.permutation(len(rows))[:n]
    cards = []
    for i in sample:
        r = rows[int(i)]
        img = cv2.imread(str(out_dir / r["cell"] / f"{r['image_id']}.jpg"))
        if img is None:
            continue
        h = 200
        img = cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            continue
        uri = "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
        cards.append(
            f'<div class="c"><img src="{uri}" alt="">'
            f'<div class="m"><span class="cell">{html.escape(r["cell"])}</span><br>'
            f'{html.escape(r["label"])}<br>'
            f'<span style="font-size:10px">{html.escape(r["image_id"])}</span><br>'
            f'bg: {html.escape(str(r["background_id"]))}</div></div>'
        )
    path.write_text(SHEET_HEAD + "\n".join(cards) + "</div>", encoding="utf-8")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def collect_sources(
    repo: Path, root_rel: str, classes: set[str] | None, reliable: set[str] | None, limit: int | None
) -> list[Path]:
    """Images under ``root_rel`` that belong to ``classes`` and have a usable leaf mask."""
    root = repo / root_rel
    if not root.exists():
        return []
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix not in IMAGE_EXTS or not p.is_file():
            continue
        if classes is not None and p.parent.name not in classes:
            continue
        if reliable is not None:
            _, rel = split_raw_path(p)
            if rel.as_posix() not in reliable:
                continue
        out.append(p)
        if limit and len(out) >= limit:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the counterfactual background-swap set.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=None, help="cap leaves per source")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    rng = np.random.default_rng(seed)

    out_dir = repo / cfg.get("out_dir", "data/counterfactual")
    leaf_root = repo / cfg.get("leaf_mask_root", "data/masks/leaf")
    lesion_root = repo / cfg.get("lesion_mask_root", "data/masks/lesion_human")
    n_bg = int(cfg.get("n_backgrounds", 40))
    feather = int(cfg.get("feather_px", 4))
    jitter = float(cfg.get("jitter", 0.15))
    limit = args.limit or cfg.get("limit_per_source")
    size = int(cfg.get("out_size", 256))

    cmap = pd.read_csv(repo / cfg.get("class_map", "data/class_map.csv"))
    crop = cfg.get("crop", "tomato")
    sel = cmap[cmap.crop.str.lower() == str(crop).lower()]
    pv_classes = set(sel.plantvillage_name.dropna().astype(str))
    pd_classes = set(sel.plantdoc_name.dropna().astype(str))

    lab_reliable = reliable_ids(leaf_root / "plantvillage" / "_leaf_mask_log.csv")
    field_reliable = reliable_ids(leaf_root / "plantdoc" / "_leaf_mask_log.csv")

    lab = collect_sources(repo, "data/raw/plantvillage/color", pv_classes, lab_reliable, limit)
    field = collect_sources(repo, "data/raw/plantdoc", pd_classes, field_reliable, limit)
    print(f"lab leaves: {len(lab)} | field leaves: {len(field)}", flush=True)
    if not lab:
        raise ValueError(
            "No lab source leaves. Run masks/leaf_masks.py on PlantVillage first "
            "(and check the crop filter in the config)."
        )

    # --- background banks -------------------------------------------------
    def build_bank(sources: list[Path], name: str) -> list[tuple[str, np.ndarray]]:
        bank = []
        for p in sources[: min(n_bg, len(sources))]:
            img = read_rgb(p)
            lm = mask_path_for(p, leaf_root)
            if not lm.exists():
                continue
            leaf = read_mask(lm, img.shape[:2])
            plate = inpaint_background(img, leaf)
            bank.append((f"{name}:{p.stem}", cv2.resize(plate, (size, size))))
        return bank

    plain_bank = build_bank(list(rng.permutation(lab)), "plain")
    field_bank = build_bank(list(rng.permutation(field)), "field") if field else []
    texture_bank = [
        (f"texture:{i}", texture_background(rng, size, size))
        for i in range(int(cfg.get("n_textures", 12)))
    ]
    print(
        f"backgrounds: plain={len(plain_bank)} field={len(field_bank)} "
        f"texture={len(texture_bank)}",
        flush=True,
    )
    if not plain_bank:
        raise ValueError("Plain background bank is empty - no usable PlantVillage leaf masks.")
    if not field_bank:
        print(
            "WARNING: field background bank is empty. The lab_field and field_field cells "
            "cannot be built; only lab_plain and paste_control will be produced.",
            flush=True,
        )

    rows: list[dict] = []

    def emit(src: Path, cell: str, bank: list[tuple[str, np.ndarray]] | None, source_ds: str):
        img = read_rgb(src)
        lm = mask_path_for(src, leaf_root)
        if not lm.exists():
            return
        leaf = read_mask(lm, img.shape[:2])
        if leaf.sum() == 0:
            return

        lesion = None
        lp = mask_path_for(src, lesion_root)
        if lp.exists():
            lesion = read_mask(lp, img.shape[:2])

        if cell == "paste_control":
            bg = cv2.resize(inpaint_background(img, leaf), (size, size))
            bg_id = f"self:{src.stem}"
        else:
            if not bank:
                return
            bg_id, bg = bank[int(rng.integers(len(bank)))]
            bg = bg.copy()

        comp, moved_leaf, moved_lesion = composite(
            img, leaf, bg, rng, feather_px=feather, jitter=jitter, lesion_mask=lesion
        )

        image_id = f"{source_ds}__{src.parent.name}__{src.stem}".replace(" ", "_")
        cdir = out_dir / cell
        cdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cdir / f"{image_id}.jpg"), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

        # Masks travel with the composite so relevance mass is computable on it.
        mdir = out_dir / "masks" / cell
        (mdir / "leaf").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(mdir / "leaf" / f"{image_id}.png"), moved_leaf * 255)
        if moved_lesion is not None:
            (mdir / "lesion_human").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mdir / "lesion_human" / f"{image_id}.png"), moved_lesion * 255)

        rows.append(
            {
                "image_id": image_id,
                "cell": cell,
                "source_dataset": source_ds,
                "label": src.parent.name,
                "background_id": bg_id,
                "has_lesion_mask": int(moved_lesion is not None),
            }
        )

    for src in lab:
        emit(src, "lab_plain", plain_bank, "plantvillage")
        emit(src, "lab_field", field_bank, "plantvillage")
        emit(src, "paste_control", None, "plantvillage")
    for src in field:
        emit(src, "field_plain", plain_bank, "plantdoc")
        emit(src, "field_field", field_bank, "plantdoc")

    if not rows:
        raise ValueError("No composites were produced - check leaf masks and the crop filter.")

    out_dir.mkdir(parents=True, exist_ok=True)
    idx = out_dir / "index.csv"
    with open(idx, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "image_id",
                "cell",
                "source_dataset",
                "label",
                "background_id",
                "has_lesion_mask",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    counts = pd.DataFrame(rows).cell.value_counts().to_dict()
    print("\ncomposites per cell:")
    for c in CELLS:
        print(f"  {c:15s} {counts.get(c, 0)}")
    print(f"\nindex: {idx}")

    sheet = out_dir / "contact_sheet.html"
    write_contact_sheet(rows, out_dir, sheet, int(cfg.get("sheet_n", 50)), rng)
    print(f"contact sheet: {sheet}")


if __name__ == "__main__":
    main()
