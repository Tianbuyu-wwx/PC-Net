# Baselines

Compact, self-contained re-implementations of the deep forecasting baselines compared in the
paper. Every file depends on `torch` only and shares the same protocol as PC-Net: a 256-step
normalised context, a 64-step horizon, and (except DLinear) a nine-quantile output head trained
with the pinball loss. They are the modules that give the parameter counts of Table 6, and
`train.py` below trains and evaluates them under the shared protocol.

| File | Model | Output | Params | Reference implementation |
|---|---|---:|---:|---|
| `lstm.py` | LSTM (2 layers, 128 hidden) | 64 × 9 | 0.273 M | PyTorch `nn.LSTM` (Hochreiter & Schmidhuber, 1997) |
| `dlinear.py` | DLinear (kernel 25) | 64 (point only) | 0.033 M | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) |
| `nbeats.py` | N-BEATS (generic, 8 stacks × 4 layers × 256) | 64 × 9 | 2.801 M | [ElementAI/N-BEATS](https://github.com/ElementAI/N-BEATS) |
| `transformer.py` | Vanilla Transformer (encoder–decoder, label length 48) | 64 × 9 | 0.538 M | [Time-Series-Library](https://github.com/thuml/Time-Series-Library) |
| `patchtst.py` | PatchTST (patch 16, stride 8, 3 layers) | 64 × 9 | 1.568 M | [yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST) |
| `itransformer.py` | iTransformer (variate tokens, 2 layers) | 64 × 9 | 1.268 M | [thuml/iTransformer](https://github.com/thuml/iTransformer) |

Quick check of the output shapes:

```python
import torch
from lstm import LSTMModel
from patchtst import PatchTSTModel
from nbeats import NBeatsModel

LSTMModel()(torch.randn(4, 256, 1)).shape        # torch.Size([4, 64, 9])
PatchTSTModel()(torch.randn(4, 256, 1)).shape    # torch.Size([4, 64, 9])
NBeatsModel()(torch.randn(4, 256)).shape         # torch.Size([4, 64, 9])
```

## Training and evaluation

`train.py` trains any of these models under the protocol of the paper and writes the same
per-step layout as PC-Net, so the produced forecasts can be scored with the same code:

```shell
python baseline/train.py --model patchtst --data data/taihu_4h_S1_yuexiqiao.parquet
python baseline/train.py --model all      --data data/taihu_4h_S1_yuexiqiao.parquet
python baseline/train.py --model lstm --test_only --ckpt outputs/baseline/ckpt/lstm.pt
```

- **Same windows as PC-Net.** Windows, z-score statistics and the chronological split all come
  from `dataset.py` of the main model, so each baseline sees the same 256-step contexts, the same
  64-step targets and the same eight test origins.
- **Same objective.** Except DLinear (squared error, point output), every model is trained with
  the nine-quantile pinball loss on the outputs described above.
- **Defaults.** AdamW with `lr = 1e-4` and weight decay `1e-5`, a cosine schedule, gradient-norm
  clip 1.0, batch size 32, 10 epochs, seed 2021 — the harness settings of this study; every
  flag can be overridden (`--epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--seed`).
- **Outputs** in `--out` (default `outputs/baseline`): `ckpt/<model>.pt`, `<model>_per_step.csv`
  (`split, section, variable, anchor, step, y_true, q10…q90, method`), `<model>_cells.csv`
  (MAE / RMSE / R² / nQS₉ / nMAE per station–indicator cell) and `summary.json`.
- DLinear only fills `q50`; its quantile columns stay `NaN`, so nQS₉ is reported as undefined —
  the same convention as in the paper.

The raw quantiles of the baselines are written as they are predicted (no monotone
rearrangement), which is why re-scoring the archived forecasts reproduces the table values
exactly while crossed quantiles remain visible in the CSVs.

## Not redistributed here

Four Transformer-family baselines and the five foundation models were run from their official
releases rather than from local re-implementations, so no source code is copied into this
repository:

- TimesNet, Autoformer, FEDformer, Informer — official [Time-Series-Library](https://github.com/thuml/Time-Series-Library),
  [Autoformer](https://github.com/thuml/Autoformer), [FEDformer](https://github.com/MAZiqing/FEDformer) and
  [Informer2020](https://github.com/zhouhaoyi/Informer2020) code, used with the same input/output protocol.
- TimesFM-2.5 / 3.0, Chronos-2, Chronos-Bolt, LakeFM — released checkpoints from
  [google-research/timesfm](https://github.com/google-research/timesfm),
  [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) and
  [KGML-lab/LakeFM](https://github.com/KGML-lab/LakeFM), all used with the same input/output protocol.

## License

The files in this folder are original re-implementations written for this study and are released
under the MIT license of this repository. The upstream projects linked above keep their own
licenses (MIT / Apache-2.0); no upstream source is redistributed here.
