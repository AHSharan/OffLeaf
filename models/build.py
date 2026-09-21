"""Model builders.

Exposes a single entry point, :func:`build_model`, returning a backbone with a
single linear head plus the extra outputs the rest of the repo depends on:

* ResNet: ``forward`` also returns the **layer-3** feature map (14x14 at 224
  input), which is what the CAM penalty in ``train/train.py`` differentiates
  through.
* ViT: per-block attention maps are retained (with gradients) so
  ``explain/methods.py`` can compute Chefer relevance.

Only ``resnet50`` and ``vit_small_patch16_224`` are supported, per spec.
"""

from __future__ import annotations

from typing import Literal

import timm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

SUPPORTED_MODELS = ("resnet50", "vit_small_patch16_224")

ModelName = Literal["resnet50", "vit_small_patch16_224"]


class ResNetWithLayer3(nn.Module):
    """ResNet wrapper whose forward returns ``(logits, layer3_features)``.

    The layer-3 map is returned rather than hooked so that it is a genuine node
    in the autograd graph of *this* forward pass. ``train/train.py`` calls
    ``autograd.grad(score, feat3, create_graph=True)`` on it, which requires the
    tensor to be part of the graph, not a detached copy captured by a hook.
    """

    def __init__(self, name: str = "resnet50", num_classes: int = 38, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        m = self.backbone
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.act1(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        feat3 = m.layer3(x)  # B x 1024 x 14 x 14 at 224 input
        x = m.layer4(feat3)

        x = m.global_pool(x)
        logits = m.fc(x)
        return logits, feat3

    @property
    def cam_target_layer(self) -> nn.Module:
        """Layer used as the Grad-CAM / HiResCAM target (spec: layer3 default)."""
        return self.backbone.layer3


class _AttentionRecorder(nn.Module):
    """Wraps a timm ``Attention`` module and retains its attention map.

    timm's fused (SDPA) attention path never materialises the attention matrix,
    so it is disabled here. This costs memory and speed, and is therefore only
    enabled when ``record_attention`` is requested.
    """

    def __init__(self, attn: nn.Module):
        super().__init__()
        self.attn = attn
        self.attn_map: Tensor | None = None
        if hasattr(attn, "fused_attn"):
            attn.fused_attn = False

    def forward(self, x: Tensor) -> Tensor:
        a = self.attn
        B, N, C = x.shape
        qkv = a.qkv(x).reshape(B, N, 3, a.num_heads, C // a.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = a.q_norm(q), a.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * a.scale
        attn = attn.softmax(dim=-1)

        # Retain for Chefer relevance: needs both the value and its gradient.
        self.attn_map = attn
        if attn.requires_grad:
            attn.retain_grad()

        attn = a.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = a.proj(x)
        x = a.proj_drop(x)
        return x


class ViTWithAttention(nn.Module):
    """ViT wrapper exposing per-block attention maps and their gradients."""

    def __init__(
        self,
        name: str = "vit_small_patch16_224",
        num_classes: int = 38,
        pretrained: bool = True,
        record_attention: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
        self.num_classes = num_classes
        self.record_attention = record_attention
        self._recorders: list[_AttentionRecorder] = []
        if record_attention:
            self.enable_attention_recording()

    def enable_attention_recording(self) -> None:
        """Swap each block's attention for a recording version (idempotent)."""
        if self._recorders:
            return
        for blk in self.backbone.blocks:
            rec = _AttentionRecorder(blk.attn)
            blk.attn = rec
            self._recorders.append(rec)
        self.record_attention = True

    @property
    def attention_maps(self) -> list[Tensor]:
        """Per-block attention, shape ``B x heads x tokens x tokens`` each."""
        if not self._recorders:
            raise RuntimeError(
                "Attention was not recorded. Build the model with record_attention=True "
                "or call enable_attention_recording() before the forward pass."
            )
        maps = [r.attn_map for r in self._recorders]
        if any(m is None for m in maps):
            raise RuntimeError("No attention maps stored yet - run a forward pass first.")
        return maps  # type: ignore[return-value]

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns ``(logits, patch_tokens)``.

        ``patch_tokens`` is reshaped to ``B x C x H x W`` so the same CAM code
        path works for both backbones.

        The head is deliberately re-fed from ``fmap`` rather than from ``feats``.
        Reshaping ``feats`` into ``fmap`` and then calling the head on ``feats``
        would leave the two as *siblings* in the autograd graph, and
        ``autograd.grad(score, fmap)`` raises "One of the differentiated Tensors
        appears to not have been used in the graph". Rebuilding the token
        sequence from ``fmap`` puts it on the path to the logits, which is what
        Grad-CAM and the training penalty both require. The round trip is a pure
        reshape/transpose, so it changes no values.
        """
        m = self.backbone
        feats = m.forward_features(x)  # B x N x C (prefix tokens included)

        n_prefix = m.num_prefix_tokens
        patches = feats[:, n_prefix:, :]  # B x (H*W) x C
        B, N, C = patches.shape
        hw = int(N**0.5)
        if hw * hw != N:
            raise ValueError(f"Non-square patch grid: {N} tokens")
        fmap = patches.transpose(1, 2).reshape(B, C, hw, hw)

        patches_back = fmap.reshape(B, C, N).transpose(1, 2)
        feats_for_head = torch.cat([feats[:, :n_prefix, :], patches_back], dim=1)
        logits = m.forward_head(feats_for_head)
        return logits, fmap


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
    record_attention: bool = False,
) -> nn.Module:
    """Build a supported backbone with a single linear head.

    Args:
        name: One of :data:`SUPPORTED_MODELS`.
        num_classes: Size of the classification head.
        pretrained: Load ImageNet weights.
        record_attention: ViT only. Retain per-block attention for Chefer
            relevance. Costs memory, so it is off during training.

    Returns:
        A module whose ``forward(x)`` returns ``(logits, feature_map)``.

    Raises:
        ValueError: If ``name`` is not supported.
    """
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {name!r}. Supported: {SUPPORTED_MODELS}")

    if name == "resnet50":
        if record_attention:
            raise ValueError("record_attention is ViT-only; ResNet uses Grad-CAM on layer3.")
        return ResNetWithLayer3(name, num_classes=num_classes, pretrained=pretrained)

    return ViTWithAttention(
        name, num_classes=num_classes, pretrained=pretrained, record_attention=record_attention
    )


def normalized_cam(feat: Tensor, grad: Tensor, out_hw: tuple[int, int] | None = None) -> Tensor:
    """Grad-CAM from a feature map and its gradient, min-max normalised per image.

    Shared by ``train/train.py`` (the penalty) and ``explain/`` so the two can
    never drift apart.

    Args:
        feat: ``B x C x H x W`` feature map.
        grad: Gradient of the target score w.r.t. ``feat``, same shape.
        out_hw: If given, bilinearly upsample to this ``(H, W)``.

    Returns:
        ``B x H x W`` map in ``[0, 1]``.
    """
    weights = grad.mean(dim=(2, 3), keepdim=True)  # B x C x 1 x 1
    cam = F.relu((weights * feat).sum(dim=1))  # B x H x W

    B = cam.shape[0]
    flat = cam.reshape(B, -1)
    lo = flat.min(dim=1).values.reshape(B, 1, 1)
    hi = flat.max(dim=1).values.reshape(B, 1, 1)
    cam = (cam - lo) / (hi - lo + 1e-8)

    if out_hw is not None:
        cam = F.interpolate(cam.unsqueeze(1), size=out_hw, mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
    return cam
