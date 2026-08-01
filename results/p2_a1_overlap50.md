# 分离评测结果：htdemucs

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | auto · htdemucs · overlap=0.5 |
| 日期 | 2026-08-01 11:01 |
| commit | `c704fa6` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `/Users/baiwenbin/音乐理解与智能混音助手/scripts/run_separation_eval.py --model htdemucs --subset test --workers 3 --overlap 0.5 --out results/p2_a1_overlap50.md` |
| 分离 RTF（中位数） | 0.0840（×11.9 实时） |
| 总耗时 | 24.5 min（音频总长 207.8 min，含 museval，3 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.92 | 10.15 | 9.83 | 6.48 | **8.84** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 15.36 | 17.04 | 8.61 | 13.18 | **13.55** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 14.85 | 16.71 | 17.35 | 8.14 | **14.26** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 6.90 | 8.61 | 6.54 | 4.70 | **6.69** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 8.70 | 10.00 | 8.76 | 6.23 | **8.42** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| htdemucs | 7.86 | 9.52 | 7.94 | 4.81 | **7.53** |
