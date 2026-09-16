"""Spec section 10: the penalty must be inert at lam=0 and effective at lam=1.

Two failure modes these guard against:

* ``lam=0`` quietly changing training anyway, which would make the dose-response
  curve's zero point incomparable to the baseline it is supposed to match.
* The penalty having no gradient path to the weights, so the intervention
  appears to run but changes nothing - the exact shape of a false null result.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from common import set_seed
from models.build import build_model, normalized_cam
from train.train import offleaf_loss

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _fixed_batch(device, b=4, n_classes=3, size=224):
    """One deterministic batch with a centred square mask."""
    g = torch.Generator(device="cpu").manual_seed(0)
    images = torch.randn(b, 3, size, size, generator=g).to(device)
    labels = torch.randint(0, n_classes, (b,), generator=g).to(device)
    mask = torch.zeros(b, size, size)
    mask[:, size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = 1.0
    return images, labels, mask.to(device)


def _penalty_step(model, images, labels, mask):
    """Forward + CAM penalty in fp32. Returns (ce, penalty)."""
    logits, feat = model(images)
    ce = F.cross_entropy(logits, labels)
    s = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    grad = torch.autograd.grad(s.sum(), feat, create_graph=True)[0]
    cam = normalized_cam(feat.float(), grad.float(), out_hw=mask.shape[-2:])
    return ce, offleaf_loss(cam, mask)


@pytest.mark.parametrize("steps", [1, 3])
def test_lambda_zero_equals_baseline(device, steps):
    """cam_penalty at lam=0 must match the baseline weights exactly.

    The two arms genuinely differ in what they compute: the lam=0 arm still runs
    ``autograd.grad(..., create_graph=True)`` and builds the CAM, then multiplies
    the result by zero. If that path perturbed anything - an extra RNG draw, a
    stray gradient contribution - the zero point of the E2 dose-response curve
    would not be comparable to E1a, and the whole curve would be misread.
    """

    def run(with_penalty: bool):
        set_seed(0, deterministic=True)
        model = build_model("resnet50", num_classes=3, pretrained=False).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        images, labels, mask = _fixed_batch(device)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            if with_penalty:
                ce, penalty = _penalty_step(model, images, labels, mask)
                loss = ce + 0.0 * penalty
            else:
                logits, _feat = model(images)
                loss = F.cross_entropy(logits, labels)
            loss.backward()
            opt.step()
        return [p.detach().cpu().clone() for p in model.parameters()]

    baseline = run(with_penalty=False)
    lam_zero = run(with_penalty=True)

    for i, (a, b) in enumerate(zip(baseline, lam_zero)):
        assert torch.allclose(a, b, atol=1e-6), (
            f"parameter {i} differs after {steps} step(s): cam_penalty at lam=0 is not "
            f"equivalent to baseline (max abs diff {(a - b).abs().max():.2e})"
        )


def test_penalty_has_gradient(device):
    """At lam=1 the penalty reaches the weights and is minimisable."""
    set_seed(0, deterministic=True)
    model = build_model("resnet50", num_classes=3, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    images, labels, mask = _fixed_batch(device)

    # --- one step: the penalty must have a non-zero gradient wrt the weights ---
    opt.zero_grad(set_to_none=True)
    _ce, penalty = _penalty_step(model, images, labels, mask)
    grads = torch.autograd.grad(
        penalty, [p for p in model.parameters() if p.requires_grad], allow_unused=True
    )
    grad_norm = float(
        torch.sqrt(sum((g.float() ** 2).sum() for g in grads if g is not None))
    )
    assert np.isfinite(grad_norm), f"penalty_grad_norm is not finite ({grad_norm})"
    assert grad_norm > 0, "penalty has no gradient path to the weights"

    # --- 50 steps on one fixed batch: the penalty must go down ---
    set_seed(0, deterministic=True)
    model = build_model("resnet50", num_classes=3, pretrained=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    history = []
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        ce, penalty = _penalty_step(model, images, labels, mask)
        (ce + 1.0 * penalty).backward()
        opt.step()
        history.append(float(penalty.detach()))

    first, last = np.mean(history[:5]), np.mean(history[-5:])
    assert last < first, (
        f"penalty did not decrease over 50 steps on a fixed batch "
        f"(first 5 mean {first:.4f} -> last 5 mean {last:.4f})"
    )


def test_penalty_is_zero_when_cam_inside_mask(device):
    """Sanity: with an all-ones mask there is no 'outside', so the penalty is 0."""
    cam = torch.rand(2, 16, 16, device=device)
    ones = torch.ones(2, 16, 16, device=device)
    assert float(offleaf_loss(cam, ones)) == pytest.approx(0.0, abs=1e-7)

    zeros = torch.zeros(2, 16, 16, device=device)
    assert float(offleaf_loss(cam, zeros)) == pytest.approx(float(cam.mean()), abs=1e-6)
