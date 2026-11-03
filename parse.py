# -*- coding: utf-8 -*-
"""Command-line arguments of PC-Net (defaults = the paper configuration)."""
import argparse


def get_args():
    parser = argparse.ArgumentParser("PC-Net")

    # ---- run control ----
    parser.add_argument("--name", type=str, default="pcnet", help="tag of this run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id (-1 for CPU)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--test_only", action="store_true", help="evaluate a saved checkpoint")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint for --test_only")
    parser.add_argument("--data", type=str, default=None, help="override gol.DATA_PATH")

    # ---- optimisation (Section 4.4) ----
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--dp", type=float, default=0.1, help="dropout on read-out features")
    parser.add_argument("--lam_F", type=float, default=0.1, help="weight of the free energy")
    parser.add_argument("--loss", type=str, default="pinball", choices=["pinball", "mse"])
    parser.add_argument("--ema", type=float, default=0.0, help="EMA decay (0 disables)")
    parser.add_argument("--amp", type=int, default=0,
                        help="bf16 autocast on CUDA (default 0 keeps full precision, as in the paper)")

    # ---- architecture ----
    parser.add_argument("--layers", type=int, default=3, help="pyramid depth L")
    parser.add_argument("--tinf", type=int, default=6, help="free-energy evaluations T_inf")
    parser.add_argument("--d", type=int, default=16, help="hidden width")
    parser.add_argument("--eta", type=float, default=0.15, help="latent update step")
    parser.add_argument("--coupling", type=str, default="bidir", choices=["bidir", "td", "bu"])
    parser.add_argument("--scales", type=str, default="pyramid", choices=["pyramid", "same"])
    parser.add_argument("--readout", type=str, default="c2f", choices=["c2f", "single"])
    parser.add_argument("--prec", type=int, default=1, help="learnable precision weights")
    parser.add_argument("--trend_feat", type=int, default=0)
    parser.add_argument("--cal_fut", type=int, default=0)

    return parser.parse_args()
