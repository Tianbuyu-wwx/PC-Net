# -*- coding: utf-8 -*-
"""LSTM baseline used in the comparison of the paper (single-variate input).

Every baseline in this folder is a compact re-implementation written for this
study, so that all models share the same input window (256 steps), the same
64-step horizon and the same nine-quantile output head.  Only the architecture
follows the cited reference; see ``baseline/README.md`` for the source links.
"""
import torch
import torch.nn as nn

NQ = 9


class LSTMModel(nn.Module):
    """Two-layer LSTM over the normalised context window."""

    def __init__(self, hidden=128, layers=2, out=64, drop=0.1, nq=NQ):
        super().__init__()
        self.out, self.nq = out, nq
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True, dropout=drop)
        self.head = nn.Linear(hidden, out * nq)

    def forward(self, x):                       # (B, T, 1)
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :]).view(-1, self.out, self.nq)
