# -*- coding: utf-8 -*-
"""PC-Net: hierarchical predictive-coding network for time-series forecasting.

The model is a direct architectural reading of predictive coding (Rao & Ballard,
1999; Friston's free energy) applied to sequence forecasting:

* **Multiresolution state pyramid.** Layer ``l`` has temporal resolution
  ``ctx / 2**l``; deeper layers therefore integrate over longer time constants.
* **Bidirectional dynamics.** Top-down prediction ``p_l = T_l(h_{l+1})``,
  error ``eps_l = h_l - p_l``; the bottom-up error is fed back through ``U_l``.
* **Inference = free-energy descent.** ``F = sum_l w_l * ||eps_l||^2`` is
  minimised for ``T_inf`` steps with a fixed step size ``eta`` and shared
  operators - a finite unrolled inference, not an unrolled network.
* **Read-out = coarse-to-fine.** Every level contributes an additive point
  forecast ``dec_l(h_l)``; a residual quantile head predicts nine offsets.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import Down2, Same3, Up2


class PCNet(nn.Module):
    """Hierarchical predictive-coding forecaster.

    Args:
        ctx: context length (256 in the paper).
        hor: forecast horizon (64 in the paper).
        layers: number of pyramid levels ``L`` (3 in the paper).
        tinf: number of free-energy evaluations ``T_inf`` (6 in the paper).
        d: hidden width (16 in the paper).
        eta: latent-state update step size (0.15 in the paper).
        pyramid: multiscale pyramid (True) or same-resolution control (False).
        coupling: ``bidir`` | ``td`` (top-down only) | ``bu`` (bottom-up only).
        lam_f: weight of the free-energy term in the training loss.
        prec: learnable precision weights ``w_l``.
        readout: ``c2f`` (additive coarse-to-fine) or ``single`` (top layer only).
        n_vars: number of variable ids (variable-conditioned state embedding).
        n_cal: calendar covariate channels (6).
        q9: number of quantiles (9).
        dropout: dropout on the read-out features.
        trend_feat: append an OLS-slope channel to the encoder input.
        cal_fut: also condition the read-out on the *future* calendar block.
    """

    def __init__(self, ctx: int = 256, hor: int = 64, layers: int = 3, tinf: int = 6,
                 d: int = 16, eta: float = 0.15, pyramid: bool = True,
                 coupling: str = "bidir", lam_f: float = 0.1, prec: bool = True,
                 readout: str = "c2f", n_vars: int = 6, n_cal: int = 6, q9: int = 9,
                 dropout: float = 0.0, trend_feat: bool = False, cal_fut: bool = False):
        """The defaults are the configuration of the paper: ``PCNet()`` has 655,203
        parameters (0.655 M) for six variables, and reproduces Table 6 of the paper."""
        super().__init__()
        assert coupling in ("bidir", "td", "bu")
        assert readout in ("c2f", "single")
        self.ctx, self.hor, self.L, self.tinf, self.eta = ctx, hor, layers, tinf, eta
        self.pyramid, self.coupling = bool(pyramid), coupling
        self.lam_f, self.prec, self.readout = lam_f, prec, readout
        self.trend_feat = bool(trend_feat)

        # temporal resolution of every pyramid level
        self.Ts = [ctx // (2 ** l) if pyramid else ctx for l in range(layers)]
        self.in_conv = nn.Conv1d(1 + n_cal + (1 if self.trend_feat else 0), d, 5, padding=2)
        # bottom-up encoder chain
        self.downs = nn.ModuleList([Down2(d) for _ in range(max(layers - 1, 0))])
        # dynamics operators: T_l (top-down prediction) and U_l (bottom-up error)
        Mk = Up2 if pyramid else Same3
        Nk = Down2 if pyramid else Same3
        self.T = nn.ModuleList([Mk(d) for _ in range(max(layers - 1, 0))])
        self.U = nn.ModuleList([Nk(d) for _ in range(max(layers - 1, 0))])
        # top-level prior
        self.prior = nn.Parameter(torch.zeros(self.Ts[-1], d))
        nn.init.normal_(self.prior, std=0.02)
        # layer norm, variable embedding, precision weights
        self.lns = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.vemb = nn.Embedding(n_vars, d)
        if prec:
            self.log_sig = nn.Parameter(torch.zeros(layers))
        # coarse-to-fine read-out: one additive decoder per level
        self.dec = nn.ModuleList([nn.Linear(self.Ts[l] * d, hor) for l in range(layers)])
        feat_dim = layers * 2 * d + hor if readout != "single" else 2 * d + hor
        self.cal_fut = bool(cal_fut)
        if self.cal_fut:
            self.calf_body = nn.Sequential(
                nn.Conv1d(n_cal, 32, 3, padding=1), nn.GELU(),
                nn.Conv1d(32, 32, 3, padding=1), nn.GELU())
            self.calf_out = nn.Conv1d(32, 1, 1)
            feat_dim += 32
        self.qhead = nn.Sequential(nn.Linear(feat_dim, 256), nn.GELU(),
                                   nn.Linear(256, hor * q9))
        self.dp = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    # dynamics
    # ------------------------------------------------------------------ #
    def _topdown(self, l, hs):
        """Top-down prediction of level ``l``: T_l(h_{l+1}), or the learned prior on top."""
        if l == self.L - 1:
            return self.prior.unsqueeze(0).expand(hs[l].shape[0], -1, -1)
        return self.T[l](hs[l + 1])

    def _w_list(self):
        """Precision weights, one per level (all ones when ``prec=False``)."""
        if self.prec:
            w = 1.0 / (F.softplus(self.log_sig) ** 2 + 1e-6)
            return [w[l] for l in range(self.L)]
        return [1.0] * self.L

    def forward(self, x, vid, cal, debug=False):
        """
        Args:
            x: (B, ctx) normalised series.
            vid: (B,) variable id.
            cal: (B, ctx + hor, n_cal) deterministic calendar covariates.

        Returns:
            q: (B, hor, q9) quantile forecasts (monotonicity enforced later).
            F_final: scalar free energy at the last inference step.
            y0: (B, hor) point forecast of the additive read-out.
        """
        B = x.shape[0]
        cal_ctx = cal[:, : self.ctx]
        chans = [x[:, :, None], cal_ctx]
        if self.trend_feat:  # global OLS slope of the context window (constant channel)
            t = torch.arange(self.ctx, device=x.device, dtype=x.dtype)
            tc = t - t.mean()
            slope = (x * tc).sum(-1, keepdim=True) / ((tc ** 2).sum() + 1e-6)
            chans.append(slope[:, :, None].expand(-1, self.ctx, 1))
        xin = torch.cat(chans, dim=-1).transpose(1, 2)     # (B, 7 or 8, ctx)
        z = self.in_conv(xin).transpose(1, 2)              # (B, ctx, d)

        # initial latent state: bottom-up encoder chain
        hs = [z]
        cur = z
        for l in range(1, self.L):
            cur = self.downs[l - 1](cur) if self.pyramid else z
            hs.append(cur)
        vem = self.vemb(vid)[:, None, :]
        hs = [self.lns[l](h + vem) for l, h in enumerate(hs)]

        w = self._w_list()

        # inference: T_inf steps of free-energy gradient descent on the latents
        F_final = None
        F_hist = []
        for t in range(self.tinf):
            eps = [hs[l] - self._topdown(l, hs) for l in range(self.L)]
            terms = [w[l] * (eps[l] ** 2).mean() for l in range(self.L)]
            Ft = sum(terms)
            F_hist.append(float(Ft.detach()))
            if t == self.tinf - 1:
                F_final = Ft
                break
            new = []
            for l in range(self.L):
                own = w[l] * eps[l]                      # own prediction error
                cross = None
                if l >= 1 and self.coupling in ("bidir", "bu"):
                    cross = w[l - 1] * self.U[l - 1](eps[l - 1])   # error rising from below
                if self.coupling == "td":
                    g = own
                elif self.coupling == "bu":
                    g = -cross if cross is not None else torch.zeros_like(own)
                else:
                    g = own - cross if cross is not None else own
                new.append(self.lns[l](hs[l] - self.eta * torch.tanh(g)))
            hs = new

        # additive coarse-to-fine read-out
        cs = [self.dec[l](hs[l].reshape(B, -1)) for l in range(self.L)]
        if self.readout == "single":
            y0 = cs[-1]
            feats = torch.cat([hs[-1].mean(1), hs[-1].std(1)], -1)
        else:
            y0 = torch.stack(cs, 0).sum(0)
            feats = torch.cat([torch.cat([h.mean(1), h.std(1)], -1) for h in hs], -1)
        if self.cal_fut:
            hcf = self.calf_body(cal[:, self.ctx:, :].transpose(1, 2))   # (B, 32, hor)
            y0 = y0 + self.calf_out(hcf).squeeze(1)
            feats = torch.cat([feats, hcf.mean(-1)], -1)
        feats = self.dp(feats)
        dq = self.qhead(torch.cat([feats, y0], -1)).view(B, self.hor, 9)
        q = y0[:, :, None] + dq
        if debug:
            return q, F_final, y0, dict(F_hist=F_hist,
                                        cs=[c.detach() for c in cs],
                                        hs=[h.detach() for h in hs])
        return q, F_final, y0
