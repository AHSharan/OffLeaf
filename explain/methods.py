"""Attribution methods. Every method returns an ``H x W`` map in ``[0, 1]``.

One signature for all of them::

    heatmap = explain(model, images, labels, method="gradcam")   # B x H x W

Grad-CAM and HiResCAM are implemented here rather than pulled from
``pytorch-grad-cam`` on purpose: the training penalty in ``train/train.py``
builds its CAM from ``models.build.normalized_cam``, and the evaluation metric
must use the *same* construction. A library implementation could differ in
normalisation or upsampling and the reported relevance mass would then measure
something subtly different from what training optimised.

Scope note: the spec also lists LRP (zennit) and Chefer relevance for ViT. Both
are omitted in this build — Chefer is ViT-only and ViT was cut from the
experiment matrix, and LRP is a second opinion there is no time to act on.
``METHODS`` is the authoritative list of what is available.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.build import normalized_cam  # noqa: E402

METHODS = ("gradcam", "hirescam")


def _minmax(cam: Tensor) -> Tensor:
    """Per-image min-max normalise a ``B x H x W`` map into ``[0, 1]``."""
    b = cam.shape[0]
    flat = cam.reshape(b, -1)
    lo = flat.min(dim=1).values.reshape(b, 1, 1)
    hi = flat.max(dim=1).values.reshape(b, 1, 1)
    return (cam - lo) / (hi - lo + 1e-8)


def _target_scores(logits: Tensor, labels: Tensor | None) -> Tensor:
    """Score to differentiate: the given class, or the predicted one."""
    if labels is None:
        labels = logits.argmax(dim=1)
    return logits.gather(1, labels.unsqueeze(1)).squeeze(1)


def gradcam(feat: Tensor, grad: Tensor, out_hw: tuple[int, int]) -> Tensor:
    """Grad-CAM: channel weights are the spatial mean of the gradient.

    Shares :func:`models.build.normalized_cam` with the training penalty, so the
    metric and the loss cannot drift apart.
    """
    return normalized_cam(feat, grad, out_hw=out_hw)


def hirescam(feat: Tensor, grad: Tensor, out_hw: tuple[int, int]) -> Tensor:
    """HiResCAM: element-wise gradient x activation, summed over channels.

    Grad-CAM averages the gradient spatially before weighting, which blurs
    fine structure. HiResCAM keeps the per-location gradient, so small lesions
    survive - which matters here, because lesions are exactly the small
    structures Grad-CAM tends to wash out.
    """
    cam = F.relu((grad * feat).sum(dim=1))
    cam = _minmax(cam)
    cam = F.interpolate(cam.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False)
    return cam.squeeze(1)


_BUILDERS: dict[str, Callable[[Tensor, Tensor, tuple[int, int]], Tensor]] = {
    "gradcam": gradcam,
    "hirescam": hirescam,
}


def explain(
    model: nn.Module,
    images: Tensor,
    labels: Tensor | None = None,
    method: str = "gradcam",
    out_hw: tuple[int, int] | None = None,
) -> Tensor:
    """Attribution map for a batch.

    Args:
        model: A model from ``models.build`` returning ``(logits, features)``.
        images: ``B x 3 x H x W``.
        labels: Class to explain. ``None`` explains the predicted class -
            which is what you want when scoring a model on data where the
            prediction, not the truth, is the thing being interrogated.
        method: One of :data:`METHODS`.
        out_hw: Output size; defaults to the input resolution.

    Returns:
        ``B x H x W`` in ``[0, 1]``.

    Raises:
        ValueError: On an unknown method.
    """
    if method not in _BUILDERS:
        raise ValueError(f"Unknown method {method!r}; available: {METHODS}")

    out_hw = out_hw or (images.shape[-2], images.shape[-1])

    # Gradients are needed even under a no_grad evaluation loop.
    with torch.enable_grad():
        images = images.detach().requires_grad_(False)
        logits, feat = model(images)
        scores = _target_scores(logits, labels)
        grad = torch.autograd.grad(scores.sum(), feat, retain_graph=False)[0]

    cam = _BUILDERS[method](feat.detach().float(), grad.detach().float(), out_hw)
    return cam.detach().clamp(0.0, 1.0)


@torch.no_grad()
def predict(model: nn.Module, images: Tensor) -> tuple[Tensor, Tensor]:
    """Return ``(predicted_class, softmax_confidence)`` for a batch."""
    logits, _ = model(images)
    probs = logits.float().softmax(dim=1)
    conf, pred = probs.max(dim=1)
    return pred, conf
