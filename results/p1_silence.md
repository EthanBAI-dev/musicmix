# 分离评测结果：silence

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | silence（全零，自检） |
| 日期 | 2026-07-30 22:17 |
| commit | `4c65b5f` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `/Users/baiwenbin/音乐理解与智能混音助手/scripts/run_separation_eval.py --model silence --subset test --workers 6 --out results/p1_silence.md` |
| 分离 RTF（中位数） | 0.0002（×5887.9 实时） |
| 总耗时 | 0.3 min（音频总长 207.8 min，含 museval，6 进程） |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| silence | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

> ⚠️ SI-SDR：全部曲目均为非有限值，无法聚合（该基线下属预期行为）。
