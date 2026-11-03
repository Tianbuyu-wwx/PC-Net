# -*- coding: utf-8 -*-
"""iTransformer baseline: each variate becomes a token.

With a single indicator per training cell (V = 1) the inverted embedding is a
linear map from the 256-step context to ``d_model``, and the encoder output at
the variate token feeds the quantile head.
"""
import torch
import torch.nn as nn

NQ = 9


class ITransformerModel(nn.Module):
    def __init__(self, ctx=256, d_model=256, nhead=8, layers=2, out=64, drop=0.1,
                 n_var=1, nq=NQ):
        super().__init__()
        self.out, self.nq = out, nq
        self.embed = nn.Linear(ctx, d_model)        # DataEmbedding_inverted
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512,
                                           dropout=drop, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.head = nn.Linear(d_model, out * nq)

    def forward(self, x):                       # (B, V, T)
        h = self.enc(self.embed(x))             # (B, V, D)
        return self.head(h).squeeze(1).view(-1, self.out, self.nq)
