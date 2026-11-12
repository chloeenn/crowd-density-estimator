"""CSRNet (Li et al., CVPR 2018): VGG-16 frontend + dilated-conv backend.

    image (B,3,H,W) -> frontend (B,512,H/8,W/8) -> backend -> density (B,1,H/8,W/8)

The predicted count of an image is `density.sum()`.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import VGG16_Weights, vgg16

# vgg16().features[:23] is exactly the first 10 conv layers (through conv4_3 + ReLU):
# 512 channels at stride 8. Verified against torchvision's layer list.
FRONTEND_END = 23
BACKEND_CHANNELS = (512, 512, 512, 256, 128, 64)
DILATION = 2


def _backend() -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = BACKEND_CHANNELS[0]
    for out_ch in BACKEND_CHANNELS:
        # padding == dilation keeps the spatial size for a 3x3 kernel.
        layers += [
            nn.Conv2d(in_ch, out_ch, 3, padding=DILATION, dilation=DILATION),
            nn.ReLU(inplace=True),
        ]
        in_ch = out_ch
    return nn.Sequential(*layers)


class CSRNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained: initialise the frontend from ImageNet VGG-16 weights (downloaded once
                and cached by torchvision). Use False for tests and when loading a checkpoint.
        """
        super().__init__()
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.frontend = vgg16(weights=weights).features[:FRONTEND_END]
        self.backend = _backend()
        self.output = nn.Conv2d(BACKEND_CHANNELS[-1], 1, kernel_size=1)
        self._init_new_layers()

    def _init_new_layers(self) -> None:
        """Backend and head: N(0, 0.01) weights, zero bias (the frontend keeps VGG's)."""
        for m in [*self.backend.modules(), self.output]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def set_frontend_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze the pretrained frontend (optional warmup)."""
        for p in self.frontend.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.backend(self.frontend(x)))
