# 分离评测结果：htdemucs

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | auto · htdemucs · overlap=0.25 · refine=mask(α=2.0) |
| 日期 | 2026-08-01 11:58 |
| commit | `c704fa6` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `python -m scripts.run_separation_eval.py --model htdemucs --subset test --workers 3 --refine mask --out results/p2_a3_mask.md` |
| 分离 RTF（中位数） | 0.0801（×12.5 实时） |
| 总耗时 | 24.3 min（音频总长 207.8 min，含 museval，3 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.10 | 7.51 | 6.97 | 5.59 | **7.04** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 16.15 | 14.56 | 7.32 | 12.93 | **12.74** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 15.19 | 14.11 | 13.37 | 7.63 | **12.57** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 6.51 | 6.54 | 4.09 | 4.39 | **5.38** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.17 | 8.22 | 6.55 | 5.59 | **7.13** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 7.22 | 7.46 | 5.18 | 4.00 | **5.96** |
