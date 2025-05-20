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
        features = self.backbone(x)  
        
        print(f"Backbone output shapes: {[f.shape for f in features]}")  
        
        # Use FPN if it exists  
        if self.fpn is not None:  
            features = self.fpn(features)  
            print(f"FPN output shapes: {[f.shape for f in features]}")  
        
        # Print the spatial dimensions for each feature map  
        for i, feat in enumerate(features):  
            print(f"Feature {i} spatial dimensions: {feat.shape[2:]} (HxW)")  
        
        x = self.encoder(features)  
        x = self.decoder(x, targets)  
    
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
