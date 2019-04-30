import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import DeepLabV2

class DeepLabv2_MSC(nn.Module):
    """
    DeepLabv2 with Multi-scale inputs
    """

    def __init__(self, classes, n_blocks, atrous_rates, scales):
        super(DeepLabv2_MSC, self).__init__()
        self.n_classes = len(classes)
        self.base = DeepLabV2(n_classes= self.n_classes, n_blocks = n_blocks, atrous_rates= atrous_rates)
        self.scales = scales
        
    def forward(self, x):
        # Original
        logits = self.base(x)
        _, _, H, W = logits.shape
        interp = lambda l: F.interpolate(
            l, size=(H, W), mode="bilinear", align_corners=False
        )

        # Scaled
        logits_pyramid = []
        for p in self.scales:
            h = F.interpolate(x, scale_factor=p, mode="bilinear", align_corners=False)
            logits_pyramid.append(self.base(h))

        # Pixel-wise max
        logits_all = [logits] + [interp(l) for l in logits_pyramid]
        logits_max = torch.max(torch.stack(logits_all), dim=0)[0]

        # if self.training:
        #     return [logits] + logits_pyramid + [logits_max]
        # else:
        return logits_max

