# PC-Net

![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch 2.9](https://img.shields.io/badge/PyTorch-2.9-ee4c2c.svg)
![Code](https://img.shields.io/badge/Code-python-purple)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

PC-Net: A Hierarchical Predictive Coding Network for Probabilistic Multi-horizon Water-Quality Forecasting

-  ***Variable-conditioned multiresolution encoding***: a temporal pyramid whose levels halve the resolution (`ctx / 2^l`) while a shared variable embedding shifts the latent state, so one small network covers all site-indicator pairs without per-variable heads.
-  ***Precision-weighted hierarchical energy***: every level carries a learnable precision `w_l`, and the inter-level consistency is measured by the predictive-coding free energy `F = Σ_l w_l ‖ε_l‖²`, with `ε_l = h_l − T_l(h_{l+1})`. `F` drives inference and regularises training (`λ_F = 0.1`).
-  ***Finite bidirectional error refinement***: top-down predictions and bottom-up errors mutually correct each other for a fixed number of steps (`T_inf = 6`, `η = 0.15`) through shared operators — no extra trainable parameters and no unrolled-network blow-up.
-  ***Additive decoding and residual quantile head***: each level contributes a coarse-to-fine point forecast and one shared head emits nine residual quantiles around it, giving point and probabilistic forecasts in a single 0.655 M-parameter pass.

The overall framework of our proposed PC-Net model is illustrated in **Fig_1**.
<p align="center">
<img align="middle" src="Figures/Fig_1.png" width="900"/>
</p>
<p align = "center">
<b>Figure 1. The overall framework of the proposed PC-Net model.</b>
</p>

## Methodology
Inference in PC-Net is read as free-energy descent on a latent pyramid. A convolutional encoder chain builds the initial multiresolution state `h_1, ..., h_L` from the normalised context window and the six deterministic calendar covariates; `L − 1` refinement sweeps then exchange top-down predictions `p_l = T_l(h_{l+1})` and bottom-up errors, updating every level with

`h_l ← LN(h_l − η · tanh(w_l ε_l − w_{l−1} U_l(ε_{l−1})))`

so that the settled state is a stationary point of `F`. Because `T_l` and `U_l` are shared 2× resampling operators, the refinement is parameter-free; `T_inf` therefore trades compute for accuracy without touching the model size. The settled states are decoded additively, and the residual quantile head predicts nine offsets around the additive point forecast, which is trained with the pinball loss plus `λ_F · F`.

Qualitative multi-horizon forecasts at the Xuhu Center station are shown in **Fig_9**.
<p align="center">
<img align="middle" src="Figures/Fig_9.png" width="900"/>
</p>
<p align = "center">
<b>Figure 9. Case visualization of multi-horizon forecasting at the Xuhu Center (S2).</b>
</p>

## Requirements
The code has been tested running under Python 3.12 with CUDA 13.0; a CPU-only installation works as well (inference only in practice).

The required packages are as follows:
- Python == 3.12
- torch == 2.9.0
- numpy == 2.5.3
- pandas == 2.3.3
- pyarrow == 25.0.0 (only needed to read the Parquet tables in [`data/`](data))

## Data
The records are surface-water measurements published by the China National Environmental Monitoring Center (https://www.cnemc.cn/sssj/), sampled hourly and resampled here to a strict 4-hour grid. The study covers seven stations of the Taihu basin and six indicators: water temperature (WT), dissolved oxygen (DO), electrical conductivity (EC), total phosphorus (TP), permanganate index (PI) and pH.

2021–2023 form the training partition, 2024 is reserved for validation, and 2025–2026 supplies the test data; the retained scalar counts are 257,823, 91,559 and 155,021 (504,403 in total). All models read 256 preceding steps (42.7 days) and predict the next 64 steps (10.7 days). Eight forecast origins (2025-01-01, 2025-04-01, 2025-07-01, 2025-09-01, 2025-10-01, 2026-01-01, 2026-04-01, 2026-07-01) define the test set.

Three of the seven stations are redistributed here as **processed Parquet tables** in [`data/`](data): Yuexi Bridge (S1, 越溪桥), Navigation Administration Station (S3, 航管站) and Guajingkou West (S4, 瓜泾口西). Each file holds the six indicators of that station on the strict 4-hour grid — 69,920–77,145 non-null scalar entries, 0.43–0.49 MB — with the schema

```
datetime,section,variable,value,quality,split
```

where `quality` is `0` for a clean observation, `2` for a gap filled by the interpolation policy and `-1` for a value that is still missing (`value` is then NaN), and `split` tags the train / validation / test segments. The complete seven-station table (504,403 scalar entries) is not redistributed. `gol.py` points to it through `DATA_PATH`; set that path to your local copy, or pass `--data` on the command line. Both Parquet and CSV inputs are accepted by `dataset.py`, which re-indexes every series onto the 4-hour grid and only keeps NaN-free windows.

## Running
**Attention:** please modify the dataset path `DATA_PATH = Path("data") / "taihu_4h_7stations_6vars.csv.gz"` in **[gol.py](gol.py)**, or point the run at another table with `--data`.

The shipped station tables can be used as-is for a quick end-to-end check,

```shell
python main.py --data data/taihu_4h_S1_yuexiqiao.parquet --name demo --epochs 5 --gpu 0
```

Then you can train the paper configuration with

```shell
nohup python main.py --name pcnet --epochs 100 --gpu 0 > pcnet.log 2>&1 &
```

Defaults follow Section 4.4 of the paper: `L=3`, `d=16`, `T_inf=6`, `η=0.15`, bidirectional coupling, pyramid scales, learnable precision, coarse-to-fine read-out, `λ_F=0.1`, AdamW for 100 epochs, `lr=1e-3`, weight decay 0.05, batch size 128, dropout 0.1, OneCycle with 5 % warm-up, gradient-norm clip 1.0 and stride 64. Training runs in full precision as reported in the paper; pass `--amp 1` to enable bf16 autocast on CUDA, which only speeds the run up. The checkpoint is written to `outputs/ckpt/<name>.pt`.

Ablations are plain flags, e.g.

```shell
python main.py --name abl_no_precision --prec 0
python main.py --name abl_topdown_only --coupling td
python main.py --name abl_same_scale   --scales same
python main.py --name abl_tinf1        --tinf 1
```

Evaluation of a saved checkpoint reports MAE / RMSE / R² / nQS₉ over the 42 station-indicator cells and dumps a per-step CSV:

```shell
python main.py --name pcnet --test_only --ckpt outputs/ckpt/pcnet.pt
```

## Baselines
PC-Net is compared with **15 forecasting models** from four families. All of them are evaluated on the same 42 station–indicator cells, read the same 256-step context and predict the same 64-step horizon, and are scored with the same metrics (MAE, RMSE, R² and nQS₉). Six deep baselines are re-implemented in **[`baseline/`](baseline)** under this shared protocol; the remaining Transformer variants and the foundation models are run from their official releases with their released checkpoints. In Table 6, the `Train (s/epoch)` entries of the four time-series foundation models report the cost of one fine-tuning pass over the 3,212 training windows under the same protocol, so that their adaptation cost can be compared with that of the task-specific models.

| # | Family | Model | Code / checkpoint |
|---:|---|---|---|
| 1 | RNN | LSTM | this repo — [`baseline/lstm.py`](baseline/lstm.py) |
| 2 | Linear | DLinear | [cure-lab/LTSF-Linear](https://github.com/cure-lab/LTSF-Linear) |
| 3 | MLP | N-BEATS | [ElementAI/N-BEATS](https://github.com/ElementAI/N-BEATS) |
| 4 | CNN | TimesNet | [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library) |
| 5 | Transformer | Transformer | this repo — [`baseline/transformer.py`](baseline/transformer.py) |
| 6 | Transformer | Informer | [zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020) |
| 7 | Transformer | Autoformer | [thuml/Autoformer](https://github.com/thuml/Autoformer) |
| 8 | Transformer | FEDformer | [MAZiqing/FEDformer](https://github.com/MAZiqing/FEDformer) |
| 9 | Transformer | PatchTST | this repo — [`baseline/patchtst.py`](baseline/patchtst.py) |
| 10 | Transformer | iTransformer | this repo — [`baseline/itransformer.py`](baseline/itransformer.py) |
| 11 | Foundation | TimesFM-2.5 | [google-research/timesfm](https://github.com/google-research/timesfm) |
| 12 | Foundation | TimesFM-3.0 | [google-research/timesfm](https://github.com/google-research/timesfm) |
| 13 | Foundation | Chronos-2 | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| 14 | Foundation | Chronos-Bolt | [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting) |
| 15 | Foundation | LakeFM | [KGML-lab/LakeFM](https://github.com/KGML-lab/LakeFM) |

The DL re-implementations ship with their parameter counts in [`baseline/README.md`](baseline/README.md); DLinear produces a point forecast only, hence its nQS₉ is not reported.

## Result
For a detailed analysis of the results, please read the paper. Here we mainly present the ablation, efficiency and horizon-wise analyses.

The trade-off between forward latency, accuracy and parameter count is visualised in **Fig_5**.
<p align="center">
<img align="middle" src="Figures/Fig_5.png" width="700"/>
</p>
<p align = "center">
<b>Figure 5. Trade-off between forward latency, prediction performance and parameter count.</b>
</p>

The ablation of the core modules at two stations is summarised in **Fig_3** and **Fig_4**.
<p align="center">
<img align="middle" src="Figures/Fig_3.png" width="900"/>
<img align="middle" src="Figures/Fig_4.png" width="900"/>
</p>
<p align = "center">
<b>Figures 3–4. Ablation results of the R² for six water-quality indicators at two stations.</b>
</p>

Error accumulation along the 64-step horizon is compared in **Fig_6** and **Fig_7**, and the sensitivity to the hierarchy depth is shown in **Fig_8**.

<p align="center">
<img align="middle" src="Figures/Fig_6.png" width="760"/>
<img align="middle" src="Figures/Fig_7.png" width="760"/>
<img align="middle" src="Figures/Fig_8.png" width="560"/>
</p>
<p align = "center">
<b>Figures 6–8. Horizon-wise MAE at two stations and sensitivity to the hierarchy depth.</b>
</p>

Averaged over the 42 cells, PC-Net reaches **nMAE 0.3755** and **nQS₉ 0.3017** with 0.655 M parameters, against 0.3886 / 0.3130 for the strongest baseline (Chronos-2, 119.5 M parameters) and 0.3974 / 0.3193 for TimesFM-3.0 (330.7 M parameters). The locations of the monitoring stations are shown in **Fig_2**.
<p align="center">
<img align="middle" src="Figures/Fig_2.png" width="620"/>
</p>
<p align = "center">
<b>Figure 2. Spatial distribution of monitoring stations.</b>
</p>

# Acknowledgements
The station records are collected and released by the China National Environmental Monitoring Center. We appreciate the efforts of the authors of DLinear, N-BEATS, TimesNet, the Transformer-based forecasting variants, TimesFM, Chronos and LakeFM, whose implementations served as the baselines of this study.

**We appreciate the efforts of these scholars and their excellent work!**
