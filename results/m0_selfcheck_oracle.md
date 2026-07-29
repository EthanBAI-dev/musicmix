# 分离评测结果：oracle

| 项 | 值 |
|---|---|
| 数据源 | 合成数据 ×6（每首 5.0s） |
| 模型 | `oracle` |
| 日期 | 2026-07-29 22:25 |
| commit | `(未提交)` |
| 硬件 | Apple M2 Max / mps |
| 命令 | `/Users/baiwenbin/音乐理解与智能混音助手/scripts/run_separation_eval.py --synthetic 6 --synthetic-seconds 5 --model oracle --out results/m0_selfcheck_oracle.md` |
| 总耗时 | 18.2s（音频总长 30s） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 21.47 | 9.92 | 30.77 | 17.39 | **19.89** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 24.57 | 16.90 | 32.64 | 21.48 | **23.90** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 29.92 | 13.19 | 38.68 | 23.91 | **26.43** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 23.83 | 12.14 | 29.96 | 20.14 | **21.52** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 19.87 | 9.92 | 27.27 | 16.98 | **18.51** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| oracle | 19.82 | 9.47 | 27.25 | 16.89 | **18.36** |
