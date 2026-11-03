# -*- coding: utf-8 -*-
"""Global configurations of PC-Net.

``DATA_PATH`` points to the complete seven-station processed table, which is not
redistributed; set it to your local copy, or pass ``--data`` on the command line
to use one of the three station tables shipped under ``data/``.
"""
from pathlib import Path

import numpy as np

# ---------------- paths ----------------
DATA_PATH = Path("data") / "taihu_4h_7stations_6vars.csv.gz"
OUTPUT_PATH = Path("outputs")

# ---------------- task definition ----------------
STATIONS = ["越溪桥", "胥湖心", "航管站", "瓜泾口西", "浏河闸", "千灯浦口", "312国道桥"]
VARIABLES = ["水温", "溶解氧", "电导率", "总磷", "高锰酸盐指数", "pH"]

CTX = 256       # input length  (256 x 4h = 42.7 days)
HOR = 64        # forecast horizon (64 x 4h = 10.7 days)
STEP = 64       # stride of the non-overlapping training windows

# ---------------- chronological split ----------------
TRAIN_START = "2021-01-01"
TRAIN_END = "2023-12-31 23:59:59"
VAL_END = "2024-12-31 23:59:59"

# eight forecast origins of the test set
TEST_ANCHORS = ["2025-01-01", "2025-04-01", "2025-07-01", "2025-09-01",
                "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01"]

# ---------------- probabilistic head ----------------
QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float64)
QCOLS = [f"q{int(t * 100)}" for t in QUANTILES]

# physically admissible ranges, used to clip de-normalised forecasts
PHYS_BOUNDS = {"水温": (-2.0, 45.0), "pH": (0.0, 14.0), "溶解氧": (0.0, 25.0),
               "高锰酸盐指数": (0.0, 50.0), "总磷": (0.0, 5.0), "电导率": (50.0, 3000.0)}

SEED = 2021
