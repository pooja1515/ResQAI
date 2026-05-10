from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class GradCAMResult:
    cam: torch.Tensor  # (H, W) float32 in [0, 1] on CPU
    target_class: int


def _find_last_conv(module: nn.Module) -> nn.Module:
    last_conv: nn.Module | None = None
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise ValueError("Could not find a Conv2d layer for Grad-CAM.")
    return last_conv


class GradCAM:
    """Minimal Grad-CAM implementation for torchvision models.

    Designed for research/production usage with small surface area:
    - Stores one activation tensor and one gradient tensor from a target layer
    - Computes normalized CAM for a chosen target class
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module | None = None) -> None:
        self.model = model
        self.target_layer = target_layer or _find_last_conv(model)
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        self._fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def _forward_hook(self, _module: nn.Module, _inp: tuple[torch.Tensor, ...], out: torch.Tensor):
        # out: (N, C, H, W)
        self._activations = out

    def _backward_hook(
        self,
        _module: nn.Module,
        _grad_inp: tuple[torch.Tensor | None, ...],
        grad_out: tuple[torch.Tensor | None, ...],
    ):
        # grad_out[0]: (N, C, H, W)
        self._gradients = grad_out[0]

    def generate(self, x: torch.Tensor, target_class: int) -> GradCAMResult:
        """Generate CAM for a single image tensor `x` (shape: 1x3xHxW)."""
        if x.dim() != 4 or x.size(0) != 1:
            raise ValueError("GradCAM expects input tensor with shape (1, 3, H, W).")

        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(x)
        if target_class < 0 or target_class >= logits.size(1):
            raise ValueError(f"target_class out of range: {target_class}")

        score = logits[0, target_class]
        score.backward(retain_graph=False)

        activations = self._activations
        gradients = self._gradients
        if activations is None or gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Compute weights: global-average-pool gradients across spatial dims
        # activations/gradients: (1, C, H, W)
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=False)  # (1, H, W)
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam.squeeze(0)
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return GradCAMResult(cam=cam.detach().float().cpu(), target_class=target_class)

