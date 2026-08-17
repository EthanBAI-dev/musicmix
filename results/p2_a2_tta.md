# 分离评测结果：htdemucs

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | auto · htdemucs · overlap=0.25 · TTA[identity,swap,flip] |
| 日期 | 2026-08-01 11:34 |
| commit | `c704fa6` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `python -m scripts.run_separation_eval.py --model htdemucs --subset test --workers 3 --tta identity swap flip --out results/p2_a2_tta.md` |
| 分离 RTF（中位数） | 0.1933（×5.2 实时） |
| 总耗时 | 32.7 min（音频总长 207.8 min，含 museval，3 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.94 | 10.07 | 10.29 | 6.46 | **8.94** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 15.15 | 17.00 | 8.65 | 13.18 | **13.49** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 14.91 | 16.66 | 17.41 | 8.18 | **14.29** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 6.75 | 8.76 | 6.70 | 4.64 | **6.72** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.75 | 10.01 | 8.85 | 6.25 | **8.47** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 7.89 | 9.54 | 8.03 | 4.82 | **7.57** |
