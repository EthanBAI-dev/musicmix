# 分离评测结果：htdemucs

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | auto · htdemucs · overlap=0.25 · refine=mwf(α=2.0) |
| 日期 | 2026-08-01 12:23 |
| commit | `c704fa6` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `python -m scripts.run_separation_eval.py --model htdemucs --subset test --workers 3 --refine mwf --out results/p2_a4_mwf.md` |
| 分离 RTF（中位数） | 0.0938（×10.7 实时） |
| 总耗时 | 24.2 min（音频总长 207.8 min，含 museval，3 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.44 | 8.43 | 7.23 | 5.72 | **7.46** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 16.55 | 14.81 | 7.03 | 13.12 | **12.88** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 15.63 | 15.42 | 13.79 | 7.60 | **13.11** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 6.60 | 7.38 | 5.16 | 4.76 | **5.97** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.39 | 8.65 | 5.88 | 5.72 | **7.16** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 7.48 | 7.96 | 4.66 | 4.20 | **6.07** |
