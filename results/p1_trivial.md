# 分离评测结果：trivial

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | trivial（混音当每一轨，下界） |
| 日期 | 2026-07-30 07:54 |
| commit | `4c65b5f` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `/Users/baiwenbin/音乐理解与智能混音助手/scripts/run_separation_eval.py --model trivial --subset test --workers 6 --out results/p1_trivial.md` |
| 分离 RTF（中位数） | 0.0002（×5808.6 实时） |
| 总耗时 | 14.9 min（音频总长 207.8 min，含 museval，6 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -5.44 | -4.21 | -6.64 | -5.07 | **-5.34** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | 30.65 | 31.70 | 17.84 | 29.03 | **27.31** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -5.45 | -4.26 | -4.60 | -5.11 | **-4.86** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | 67.46 | 67.46 | 67.46 | 67.46 | **67.46** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -6.63 | -3.87 | -6.65 | -5.09 | **-5.56** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -6.66 | -3.92 | -6.67 | -5.15 | **-5.60** |
