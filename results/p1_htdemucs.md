# 分离评测结果：htdemucs

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | htdemucs（auto, overlap=0.25） |
| 日期 | 2026-07-30 10:39 |
| commit | `4c65b5f` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `/Users/baiwenbin/音乐理解与智能混音助手/scripts/run_separation_eval.py --model htdemucs --subset test --workers 4 --out results/p1_htdemucs.md` |
| 分离 RTF（中位数） | 0.0660（×15.2 实时） |
| 总耗时 | 18.6 min（音频总长 207.8 min，含 museval，4 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.93 | 10.05 | 9.78 | 6.42 | **8.80** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 15.10 | 17.08 | 8.38 | 13.16 | **13.43** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 14.91 | 16.65 | 17.36 | 8.19 | **14.28** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 6.78 | 8.71 | 6.62 | 4.62 | **6.68** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.70 | 9.94 | 8.71 | 6.19 | **8.38** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 7.83 | 9.44 | 7.87 | 4.76 | **7.48** |
