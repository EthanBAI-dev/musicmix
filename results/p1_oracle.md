# 分离评测结果：oracle

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | oracle（IRM 理想掩码，上界） |
| 日期 | 2026-07-30 08:09 |
| commit | `4c65b5f` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `python -m scripts.run_separation_eval.py --model oracle --subset test --workers 6 --out results/p1_oracle.md` |
| 分离 RTF（中位数） | 0.0332（×30.2 实时） |
| 总耗时 | 14.6 min（音频总长 207.8 min，含 museval，6 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 10.05 | 9.30 | 7.75 | 8.69 | **8.95** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 18.44 | 16.39 | 9.66 | 15.59 | **15.02** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 18.46 | 15.43 | 14.97 | 13.20 | **15.51** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 9.60 | 8.34 | 6.48 | 8.10 | **8.13** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 11.42 | 10.42 | 8.46 | 9.12 | **9.85** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 11.07 | 9.96 | 7.65 | 8.55 | **9.31** |
