# -*- coding: utf-8 -*-
"""PatchTST baseline (patch_len 16, stride 8, 3 encoder layers).

Patches are embedded linearly, a learnable positional code is added, and the
flattened encoder output is mapped to the 64-step x 9-quantile grid.
"""
import torch
import torch.nn as nn

NQ = 9


class PatchTSTModel(nn.Module):
    def __init__(self, patch=16, stride=8, d_model=128, nhead=8, layers=3, out=64,
                 drop=0.2, ctx=256, nq=NQ):
        super().__init__()
        self.patch, self.stride, self.out, self.nq = patch, stride, out, nq
        self.n_patch = (ctx - patch) // stride + 1
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patch, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                           dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Sequential(nn.Flatten(),
                                  nn.Linear(self.n_patch * d_model, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, out * nq))

    def forward(self, x):                       # (B, T, 1)
        h = x.squeeze(-1).unfold(1, self.patch, self.stride)     # (B, n_patch, patch)
        h = self.embed(h) + self.pos
        return self.head(self.enc(h)).view(-1, self.out, self.nq)
