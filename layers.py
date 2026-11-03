# -*- coding: utf-8 -*-
"""Basic layers shared by the predictive-coding dynamics.

The two resolution operators (``Down2``/``Up2``) are the *only* operators the
top-down predictions and the bottom-up errors are built from, so the whole
bidirectional refinement adds no parameters beyond these convolutions.  The
``Same3`` operator is used by the same-resolution control (``--scales same``).
"""
from __future__ import annotations

import torch.nn as nn


class Down2(nn.Module):
    """2x temporal down-sampling (strided conv), channels d -> d."""

    def __init__(self, d: int):
        super().__init__()
        self.c = nn.Conv1d(d, d, 3, stride=2, padding=1)

    def forward(self, x):  # (B, T, d) -> (B, T/2, d)
        return self.c(x.transpose(1, 2)).transpose(1, 2)


class Up2(nn.Module):
    """2x temporal up-sampling (transposed conv), channels d -> d."""

    def __init__(self, d: int):
        super().__init__()
        self.c = nn.ConvTranspose1d(d, d, 4, stride=2, padding=1)

    def forward(self, x):  # (B, T, d) -> (B, 2T, d)
        return self.c(x.transpose(1, 2)).transpose(1, 2)


class Same3(nn.Module):
    """Same-resolution 3x1 conv; used when the pyramid is disabled."""

    def __init__(self, d: int):
        super().__init__()
        self.c = nn.Conv1d(d, d, 3, padding=1)

    def forward(self, x):  # (B, T, d) -> (B, T, d)
        return self.c(x.transpose(1, 2)).transpose(1, 2)
