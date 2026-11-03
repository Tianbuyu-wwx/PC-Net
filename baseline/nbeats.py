# -*- coding: utf-8 -*-
"""N-BEATS baseline (generic architecture, 8 stacks x 4 layers x 256).

Each block predicts a backcast that is subtracted from the residual input and a
forecast that is accumulated; the last dimension carries the nine quantiles.
"""
import torch
import torch.nn as nn

NQ = 9


class NBeatsBlock(nn.Module):
    def __init__(self, ctx, theta, horizon, hidden=256, layers=4):
        super().__init__()
        self.fc = nn.ModuleList([nn.Linear(ctx, hidden)] +
                                [nn.Linear(hidden, hidden) for _ in range(layers - 1)])
        self.tb, self.tf = nn.Linear(hidden, theta), nn.Linear(hidden, theta)
        self.backcast = nn.Linear(theta, ctx)
        self.forecast = nn.Linear(theta, horizon * NQ)

    def forward(self, x):
        h = x
        for l in self.fc:
            h = torch.relu(l(h))
        return self.backcast(self.tb(h)), self.forecast(self.tf(h))


class NBeatsModel(nn.Module):
    def __init__(self, ctx=256, horizon=64, stacks=8, layers=4, hidden=256):
        super().__init__()
        self.out = horizon
        self.stacks = nn.ModuleList([NBeatsBlock(ctx, 64, horizon, hidden, layers)
                                     for _ in range(stacks)])

    def forward(self, x):                       # (B, T)
        residual, fc = x, 0
        for s in self.stacks:
            b, f = s(residual)
            residual = residual - b
            fc = fc + f
        return fc.view(-1, self.out, NQ)
