# -*- coding: utf-8 -*-
"""Vanilla Transformer baseline (encoder-decoder, label length 48).

Follows the Time-Series-Library protocol: the decoder receives the last
``label_len`` steps as a start token and the encoder output as memory.
"""
import torch
import torch.nn as nn

NQ = 9


class TransformerModel(nn.Module):
    def __init__(self, d_model=128, nhead=8, e_layers=2, d_layers=1, out=64,
                 drop=0.1, label_len=48, nq=NQ):
        super().__init__()
        self.label_len, self.out, self.nq = label_len, out, nq
        self.enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=256,
                                       dropout=drop, batch_first=True), e_layers)
        self.dec = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=256,
                                       dropout=drop, batch_first=True), d_layers)
        self.embed = nn.Linear(1, d_model)
        self.proj = nn.Linear(d_model, out * nq)

    def forward(self, x):                       # (B, T, 1)
        h = self.enc(self.embed(x))
        d = self.dec(self.embed(x[:, -self.label_len:, :]), h)
        return self.proj(d.mean(1)).view(-1, self.out, self.nq)
