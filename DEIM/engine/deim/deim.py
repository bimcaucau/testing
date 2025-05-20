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
        
        # Use FPN if it exists  
        if self.fpn is not None:  
            # Make sure features is a list  
            if not isinstance(features, list):  
                features = [features]  
                
            # Print debug information  
            print(f"Backbone output shapes: {[f.shape for f in features]}")  
            
            try:  
                features = self.fpn(features)  
            except Exception as e:  
                print(f"Error in FPN: {e}")  
                # Don't try to access in_channels directly  
                print(f"Feature shapes: {[f.shape for f in features]}")  
                raise  
        
        x = self.encoder(features)  
        x = self.decoder(x, targets)  
    
        return x

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self
