# 分离评测结果：trivial

| 项 | 值 |
|---|---|
| 数据源 | 合成数据 ×6（每首 5.0s） |
| 模型 | `trivial` |
| 日期 | 2026-07-29 22:25 |
| commit | `(未提交)` |
| 硬件 | Apple M2 Max / mps |
| 命令 | `python -m scripts.run_separation_eval.py --synthetic 6 --synthetic-seconds 5 --model trivial --out results/m0_selfcheck_trivial.md` |
| 总耗时 | 17.1s（音频总长 30s） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -5.24 | -15.84 | 2.05 | -8.21 | **-6.81** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | 17.80 | 7.81 | 24.08 | 14.88 | **16.14** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -5.18 | -15.16 | 2.05 | -8.03 | **-6.58** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | 145.44 | 145.44 | 145.44 | 145.44 | **145.44** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -5.28 | -15.85 | 2.04 | -8.22 | **-6.83** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| trivial | -5.31 | -15.80 | 2.03 | -8.21 | **-6.82** |
