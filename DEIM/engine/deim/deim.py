"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', 'fpn', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
        fpn: nn.Module = None
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.fpn = fpn

    def forward(self, x, targets=None):
        x = self.backbone(x)

        if self.fpn is not None:
            x = self.fpn(x)

        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
