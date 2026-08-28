# 分离评测结果：silence

| 项 | 值 |
|---|---|
| 数据源 | 合成数据 ×6（每首 5.0s） |
| 模型 | `silence` |
| 日期 | 2026-07-29 22:27 |
| commit | `(未提交)` |
| 硬件 | Apple M2 Max / mps |
| 命令 | `python -m scripts.run_separation_eval.py --synthetic 6 --synthetic-seconds 5 --model silence --out results/m0_selfcheck_silence.md` |
| 总耗时 | 0.1s（音频总长 30s） |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| silence | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

> ⚠️ SI-SDR：全部曲目均为非有限值，无法聚合（该基线下属预期行为）。
