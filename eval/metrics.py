"""All reported metrics.

Naming follows the project convention: the penalty is ``offleaf_loss`` and the
two reliance metrics are **``offleaf_mass``** (attribution mass off the leaf,
i.e. on background) and **``offlesion_mass``** (attribution mass off the lesion).

The pseudo-mask refusal in :func:`relevance_mass` is a correctness rule, not a
convenience check - see CLAUDE.md section 6. Model-generated masks may train a
model; they may never measure one, because a metric computed against a
segmenter's output partly measures agreement with that segmenter rather than
with the disease.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, f1_score

BOOTSTRAP_N = 1000
PSEUDO_MARKERS = ("lesion_pseudo", "pseudo")


# --------------------------------------------------------------------------
# accuracy and uncertainty
# --------------------------------------------------------------------------


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.size == 0:
        raise ValueError("accuracy on an empty set")
    return float((y_true == y_pred).mean())


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_report(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str] | None = None
) -> dict:
    return classification_report(
        y_true,
        y_pred,
        target_names=list(class_names) if class_names else None,
        output_dict=True,
        zero_division=0,
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic=np.mean,
    n: int = BOOTSTRAP_N,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI over per-item values.

    Field test sets here are small (PlantDoc's test split is 236 images), so a
    bare accuracy number is not interpretable on its own. Every reported field
    number carries one of these.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("bootstrap on an empty set")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    stats = statistic(arr[idx], axis=1)
    return {
        "point": float(statistic(arr)),
        "lo": float(np.percentile(stats, 100 * alpha / 2)),
        "hi": float(np.percentile(stats, 100 * (1 - alpha / 2))),
        "n": int(arr.size),
        "resamples": int(n),
    }


def accuracy_with_ci(y_true: Sequence[int], y_pred: Sequence[int], seed: int = 0) -> dict:
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    return bootstrap_ci(correct, seed=seed)


# --------------------------------------------------------------------------
# relevance mass - the reliance metrics
# --------------------------------------------------------------------------


def assert_human_masks(mask_dir: str | Path, allow_pseudo: bool = False) -> None:
    """Refuse to compute a reported metric on model-generated masks.

    Raises:
        ValueError: If the path looks like a pseudo-mask directory and
            ``allow_pseudo`` was not explicitly passed.
    """
    s = str(mask_dir).replace("\\", "/").lower()
    if any(m in s for m in PSEUDO_MARKERS) and not allow_pseudo:
        raise ValueError(
            f"Refusing to compute relevance mass on pseudo masks: {mask_dir}\n"
            "These are model-generated; a metric computed against them partly measures "
            "agreement with the segmenter, not with the disease. Point this at "
            "data/masks/lesion_human/, or pass --allow_pseudo and label every resulting "
            "number as pseudo-mask-based."
        )


def _mass_fractions(
    heat: np.ndarray, leaf: np.ndarray, lesion: np.ndarray | None
) -> dict[str, float]:
    """Fractions of attribution mass on lesion / leaf-not-lesion / background."""
    total = float(heat.sum())
    if total <= 0:
        return {
            "lesion_mass": 0.0,
            "leaf_not_lesion_mass": 0.0,
            "offleaf_mass": 0.0,
            "offlesion_mass": 0.0,
            "degenerate": 1.0,
        }

    leaf_b = (leaf > 0.5).astype(np.float32)
    on_leaf = float((heat * leaf_b).sum())
    offleaf = 1.0 - on_leaf / total

    if lesion is None:
        return {
            "lesion_mass": float("nan"),
            "leaf_not_lesion_mass": float("nan"),
            "offleaf_mass": offleaf,
            "offlesion_mass": float("nan"),
            "degenerate": 0.0,
        }

    les_b = (lesion > 0.5).astype(np.float32)
    on_lesion = float((heat * les_b).sum())
    return {
        "lesion_mass": on_lesion / total,
        "leaf_not_lesion_mass": float((heat * leaf_b * (1 - les_b)).sum()) / total,
        "offleaf_mass": offleaf,
        # Everything not on the lesion - background included. This is the
        # quantity the lesion-level penalty is trying to drive down.
        "offlesion_mass": 1.0 - on_lesion / total,
        "degenerate": 0.0,
    }


def top_mass_iou(heat: np.ndarray, target: np.ndarray, frac: float = 0.5) -> float:
    """IoU between the region holding the top ``frac`` of attribution and ``target``.

    Complements the mass fractions: mass says *how much* attribution is in the
    right place, this says whether its *shape* matches.
    """
    tgt = target > 0.5
    if heat.sum() <= 0:
        return 0.0
    flat = np.sort(heat.ravel())[::-1]
    csum = np.cumsum(flat)
    cutoff_idx = int(np.searchsorted(csum, frac * csum[-1]))
    cutoff_idx = min(cutoff_idx, flat.size - 1)
    thr = flat[cutoff_idx]
    region = heat >= thr
    union = np.logical_or(region, tgt).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(region, tgt).sum() / union)


def relevance_mass(
    heatmaps: Sequence[np.ndarray],
    leaf_masks: Sequence[np.ndarray],
    lesion_masks: Sequence[np.ndarray] | None = None,
    lesion_mask_dir: str | Path | None = None,
    allow_pseudo: bool = False,
    seed: int = 0,
) -> dict:
    """Aggregate ``offleaf_mass`` / ``offlesion_mass`` with bootstrap CIs.

    Args:
        heatmaps: Per-image ``H x W`` attribution maps in ``[0, 1]``.
        leaf_masks: Per-image binary leaf masks.
        lesion_masks: Per-image binary lesion masks, or ``None`` for
            leaf-level only.
        lesion_mask_dir: Where the lesion masks came from. Checked against the
            pseudo-mask rule.
        allow_pseudo: Explicitly permit pseudo masks; the result is then tagged.

    Raises:
        ValueError: On a pseudo-mask directory without ``allow_pseudo``, or on
            mismatched input lengths.
    """
    if lesion_mask_dir is not None:
        assert_human_masks(lesion_mask_dir, allow_pseudo)
    if len(heatmaps) != len(leaf_masks):
        raise ValueError(f"length mismatch: {len(heatmaps)} heatmaps, {len(leaf_masks)} leaf masks")
    if lesion_masks is not None and len(lesion_masks) != len(heatmaps):
        raise ValueError("lesion mask count does not match heatmap count")
    if not heatmaps:
        raise ValueError("relevance_mass on an empty set")

    rows, ious, degenerate = [], [], 0
    for i, (h, lf) in enumerate(zip(heatmaps, leaf_masks)):
        les = lesion_masks[i] if lesion_masks is not None else None
        r = _mass_fractions(np.asarray(h, dtype=np.float32), np.asarray(lf), None if les is None else np.asarray(les))
        degenerate += int(r.pop("degenerate"))
        rows.append(r)
        if les is not None and np.asarray(les).sum() > 0:
            ious.append(top_mass_iou(np.asarray(h, dtype=np.float32), np.asarray(les)))

    out: dict = {
        "n_images": len(rows),
        "degenerate_heatmaps": degenerate,
        "mask_source": "pseudo" if allow_pseudo else "human",
        "PSEUDO_MASK_BASED": bool(allow_pseudo),
    }
    for key in ("offleaf_mass", "offlesion_mass", "lesion_mass", "leaf_not_lesion_mass"):
        vals = [r[key] for r in rows if not np.isnan(r[key])]
        out[key] = bootstrap_ci(vals, seed=seed) if vals else None
    out["top50_mass_iou"] = bootstrap_ci(ious, seed=seed) if ious else None
    return out


# --------------------------------------------------------------------------
# flip rate - the behavioural measure
# --------------------------------------------------------------------------


def flip_rate(
    pred_a: Sequence[int],
    conf_a: Sequence[float],
    pred_b: Sequence[int],
    conf_b: Sequence[float],
    seed: int = 0,
) -> dict:
    """How often the prediction changes when only the background changed.

    Needs no attribution method at all - it is the model's behaviour, not an
    approximation of its reasoning. That independence is the point: if this
    agrees with ``offleaf_mass``, attribution maps are trustworthy in this
    domain; if it disagrees, heatmap-based claims elsewhere are on thin ice.
    """
    a, b = np.asarray(pred_a), np.asarray(pred_b)
    if a.shape != b.shape:
        raise ValueError(f"prediction arrays differ in shape: {a.shape} vs {b.shape}")
    if a.size == 0:
        raise ValueError("flip_rate on an empty set")

    flipped = (a != b).astype(float)
    dconf = np.abs(np.asarray(conf_a, dtype=float) - np.asarray(conf_b, dtype=float))
    return {
        "flip_rate": bootstrap_ci(flipped, seed=seed),
        "mean_abs_delta_confidence": bootstrap_ci(dconf, seed=seed),
        "n_pairs": int(a.size),
    }


# --------------------------------------------------------------------------
# gap decomposition
# --------------------------------------------------------------------------


def gap_decomposition(cell_accuracy: dict[str, float]) -> dict:
    """Split the lab-to-field accuracy gap into background and leaf components.

    Expects accuracies for ``lab_plain``, ``lab_field``, ``field_plain``,
    ``field_field`` and ``paste_control``.

    ``paste_control`` is the leaf pasted back onto its own background: it
    carries every compositing artefact and no background change, so its drop
    from ``lab_plain`` is the cost of the pasting itself and is subtracted from
    the background effect. Without that correction a measured "background
    effect" could simply be the seam left by feathering.

    Raises:
        KeyError: If a required cell is missing.
    """
    required = ["lab_plain", "lab_field", "field_plain", "field_field", "paste_control"]
    missing = [c for c in required if c not in cell_accuracy]
    if missing:
        raise KeyError(f"gap_decomposition missing cell(s): {missing}")

    lab_plain = cell_accuracy["lab_plain"]
    total_gap = lab_plain - cell_accuracy["field_field"]

    paste_cost = max(lab_plain - cell_accuracy["paste_control"], 0.0)
    bg_effect_raw = lab_plain - cell_accuracy["lab_field"]
    bg_effect = max(bg_effect_raw - paste_cost, 0.0)
    leaf_effect = max(lab_plain - cell_accuracy["field_plain"] - paste_cost, 0.0)

    share = (bg_effect / total_gap) if total_gap > 1e-9 else float("nan")
    return {
        "cell_accuracy": dict(cell_accuracy),
        "total_gap": total_gap,
        "paste_artefact_cost": paste_cost,
        "background_effect": bg_effect,
        "background_effect_uncorrected": bg_effect_raw,
        "leaf_appearance_effect": leaf_effect,
        "background_share_of_gap": share,
        "note": (
            "background_effect is corrected by subtracting paste_artefact_cost. "
            "Effects are not required to sum to total_gap - they are separate "
            "interventions, and interactions between them are not measured."
        ),
    }
