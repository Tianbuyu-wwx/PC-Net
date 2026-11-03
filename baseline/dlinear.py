# -*- coding: utf-8 -*-
"""DLinear baseline: series decomposition + two linear maps (point forecast).

The paper reports no quantile score for this baseline, therefore the module
returns a single point trajectory instead of a quantile fan.
"""
import torch.nn.functional as F
import torch.nn as nn


class DLinear(nn.Module):
    def __init__(self, ctx=256, hor=64, kernel=25):
        super().__init__()
        self.k = kernel
        self.linear_trend = nn.Linear(ctx, hor)
        self.linear_seas = nn.Linear(ctx, hor)

    def _decompose(self, x):
        pad = F.pad(x.unsqueeze(1), (self.k // 2, self.k // 2), mode="replicate")
        trend = F.avg_pool1d(pad, self.k, stride=1).squeeze(1)
        return trend, x - trend

    def forward(self, x):                       # (B, T)
        tr, se = self._decompose(x)
        return self.linear_trend(tr) + self.linear_seas(se)      # (B, H)
