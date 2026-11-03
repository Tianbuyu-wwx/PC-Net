# Data

Three of the seven stations are redistributed as **processed Parquet tables**. Each file is one
station on the strict 4-hour grid, in long format, exactly as produced by our preprocessing
pipeline:

| File | Station | Span | Steps | Non-null values | Size |
|---|---|---|---:|---:|---:|
| `taihu_4h_S1_yuexiqiao.parquet` | 越溪桥 Yuexi Bridge (S1) | 2020-05-01 00:00 → 2026-09-09 16:00 | 13,937 | 77,145 | 0.43 MB |
| `taihu_4h_S3_hangguanzhan.parquet` | 航管站 Navigation Administration Station (S3) | 2020-05-01 00:00 → 2026-09-09 16:00 | 13,937 | 76,866 | 0.49 MB |
| `taihu_4h_S4_guajingkoux.parquet` | 瓜泾口西 Guajingkou West (S4) | 2021-04-20 12:00 → 2026-09-09 16:00 | 11,810 | 69,920 | 0.48 MB |

Every file carries the six indicators used in the paper — water temperature (WT), dissolved oxygen
(DO), electrical conductivity (EC), total phosphorus (TP), permanganate index (PI) and pH — with
the same schema:

| Column | Meaning |
|---|---|
| `datetime` | timestamp on the 4-hour grid |
| `section` | station name (Chinese, identical to `gol.STATIONS`) |
| `variable` | indicator name (Chinese, identical to `gol.VARIABLES`) |
| `value` | measurement in physical units; **NaN where the record is missing** |
| `quality` | `0` = clean observation, `2` = gap filled by the interpolation policy, `-1` = still missing |
| `split` | `train` (≤ 2023-12-31), `val` (2024), `test` (≥ 2025-01-01); 2020 is `drop` and only serves as warm-up context |

The remaining four stations are not redistributed; together the seven span 504,403 non-null scalar
entries over the 2021–2026 analysis window. The three files above keep the exact column names,
station names and indicator names of the full table, so the code path is identical.

## Using the sample

```shell
python main.py --data data/taihu_4h_S1_yuexiqiao.parquet --name demo --epochs 20
```

Window counts per file (8 forecast origins × 6 indicators):

| File | Train windows | Val windows | Test sequences |
|---|---:|---:|---:|
| S1 | 583 | 204 | 48 |
| S3 | 558 | 204 | 48 |
| S4 | 456 | 204 | 48 |

With the complete seven-station table the same code produces 3,212 training windows over 42
station–indicator cells.

## Using your own data

Set `DATA_PATH` in [gol.py](../gol.py) (or pass `--data`) to any `.parquet`, `.csv` or `.csv.gz`
table with the columns above. `dataset.py` re-indexes every series onto the 4-hour grid, fits the
z-score statistics on 2021–2023 only, and keeps NaN-free `context + horizon` windows, so missing
samples are tolerated. Reading Parquet requires `pyarrow`.

## Source

China National Environmental Monitoring Center, <https://www.cnemc.cn/sssj/> — hourly surface-water
records, resampled here to 4 h, screened against physical bounds, Hampel-filtered and split as
described above.
