"""Stratified PlantVillage splits, written to disk once and reused forever.

Experiments never re-split. They load the CSVs this module writes, so every
regime at a given seed sees byte-identical train/val/test membership and any
accuracy difference is attributable to the regime rather than to the split.

Usage::

    python data/splits.py --config configs/splits.yaml
    python data/splits.py --config configs/splits.yaml --seed 1
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import load_config, set_seed  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

REPO_ROOT = Path(__file__).resolve().parents[1]


def _relative_to_repo(p: Path) -> str:
    """Path relative to the repo root, posix-style, for portable split files."""
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        # Outside the repo (e.g. a dataset on another drive): keep it absolute
        # rather than emitting a broken relative path.
        return p.resolve().as_posix()


def discover_classes(source_dir: Path) -> dict[str, list[Path]]:
    """Map each class folder under ``source_dir`` to its image paths.

    Raises:
        FileNotFoundError: If ``source_dir`` does not exist.
        ValueError: If it contains no class folders.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Image root not found: {source_dir}")

    classes: dict[str, list[Path]] = {}
    for d in sorted(p for p in source_dir.iterdir() if p.is_dir()):
        imgs = sorted(p for p in d.iterdir() if p.suffix in IMAGE_EXTS and p.is_file())
        if imgs:
            classes[d.name] = imgs
    if not classes:
        raise ValueError(f"No class folders with images under {source_dir}")
    return classes


def warn_unmapped(class_names: list[str], class_map_path: Path, column: str) -> None:
    """Warn for any raw folder absent from ``class_map.csv`` (spec section 2)."""
    if not class_map_path.exists():
        warnings.warn(
            f"{class_map_path} not found - skipping the class-map check. "
            "Run data/make_class_map.py first.",
            stacklevel=2,
        )
        return
    with open(class_map_path, encoding="utf-8") as fh:
        mapped = {row[column].strip() for row in csv.DictReader(fh) if row.get(column, "").strip()}
    missing = [c for c in class_names if c not in mapped]
    if missing:
        warnings.warn(
            f"{len(missing)} folder(s) under the raw tree are not in {class_map_path.name} "
            f"(column {column!r}): {missing}",
            stacklevel=2,
        )


def make_splits(
    source_dir: Path,
    out_dir: Path,
    seed: int,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    dataset_name: str = "plantvillage",
    class_map_path: Path | None = None,
    class_map_column: str = "plantvillage_name",
) -> dict[str, Path]:
    """Write stratified train/val/test CSVs for one seed.

    Args:
        source_dir: e.g. ``data/raw/plantvillage/color``.
        out_dir: e.g. ``data/splits``.
        seed: Split seed; also the filename suffix.
        ratios: Train/val/test fractions, must sum to 1.
        dataset_name: Filename prefix.
        class_map_path: Optional ``data/class_map.csv`` to validate against.
        class_map_column: Column of ``class_map.csv`` holding this dataset's names.

    Returns:
        Mapping of split name to the CSV written.

    Raises:
        ValueError: If ratios do not sum to 1, or a class is too small to split.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios} -> {sum(ratios)}")

    classes = discover_classes(source_dir)
    class_names = sorted(classes)
    if class_map_path is not None:
        warn_unmapped(class_names, class_map_path, class_map_column)

    class_to_idx = {c: i for i, c in enumerate(class_names)}

    paths: list[Path] = []
    labels: list[int] = []
    for c in class_names:
        if len(classes[c]) < 3:
            raise ValueError(
                f"Class {c!r} has only {len(classes[c])} image(s); cannot make a "
                "stratified 3-way split. Drop the class explicitly or add images."
            )
        paths.extend(classes[c])
        labels.extend([class_to_idx[c]] * len(classes[c]))

    train_frac, val_frac, test_frac = ratios

    # First carve off train, then split the remainder into val/test.
    p_train, p_rest, y_train, y_rest = train_test_split(
        paths, labels, train_size=train_frac, stratify=labels, random_state=seed
    )
    rel_val = val_frac / (val_frac + test_frac)
    p_val, p_test, y_val, y_test = train_test_split(
        p_rest, y_rest, train_size=rel_val, stratify=y_rest, random_state=seed
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for split, ps, ys in (
        ("train", p_train, y_train),
        ("val", p_val, y_val),
        ("test", p_test, y_test),
    ):
        out = out_dir / f"{dataset_name}_seed{seed}_{split}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "label", "class_name"])
            for p, y in zip(ps, ys):
                # Repo-relative, so a split generated on one machine is usable on
                # another. Absolute paths here would hardcode this drive letter.
                w.writerow([_relative_to_repo(p), y, class_names[y]])
        written[split] = out
        print(f"  {split:5s} {len(ps):6d} images -> {out.name}")

    idx_path = out_dir / f"{dataset_name}_classes.csv"
    with open(idx_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "class_name"])
        for c in class_names:
            w.writerow([class_to_idx[c], c])
    written["classes"] = idx_path

    return written


def load_split(csv_path: str | Path) -> tuple[list[Path], list[int]]:
    """Read a split CSV back into ``(paths, labels)``.

    Raises:
        FileNotFoundError: If the split has not been generated yet.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Split not found: {p}. Generate it first with data/splits.py --config ..."
        )
    paths: list[Path] = []
    labels: list[int] = []
    with open(p, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            q = Path(row["path"])
            # Split files store repo-relative paths; resolve them here.
            paths.append(q if q.is_absolute() else REPO_ROOT / q)
            labels.append(int(row["label"]))
    return paths, labels


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate stratified dataset splits.")
    ap.add_argument("--config", required=True, help="yaml config")
    ap.add_argument("--seed", type=int, default=None, help="override config.seed")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [cfg.get("seed", 0)])

    repo = Path(__file__).resolve().parents[1]
    source_dir = repo / cfg["source_dir"]
    out_dir = repo / cfg.get("out_dir", "data/splits")
    class_map = repo / cfg.get("class_map", "data/class_map.csv")
    ratios = tuple(cfg.get("ratios", [0.8, 0.1, 0.1]))

    for seed in seeds:
        print(f"seed {seed}:")
        set_seed(seed)
        make_splits(
            source_dir=source_dir,
            out_dir=out_dir,
            seed=seed,
            ratios=ratios,  # type: ignore[arg-type]
            dataset_name=cfg.get("dataset_name", "plantvillage"),
            class_map_path=class_map,
            class_map_column=cfg.get("class_map_column", "plantvillage_name"),
        )


if __name__ == "__main__":
    main()
