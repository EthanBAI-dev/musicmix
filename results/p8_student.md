# 分离评测结果：student

| 项 | 值 |
|---|---|
| 数据源 | MUSDB18-HQ test（50 首） |
| 模型 | student（蒸馏，3.11 M 参数，step 20000） |
| 日期 | 2026-09-06 18:59 |
| commit | `4c5ce8f` |
| 硬件 | Apple M2 Max / torch 2.13.0 |
| 命令 | `/Users/baiwenbin/音乐理解与智能混音助手/scripts/run_separation_eval.py --model student --ckpt results/student_final.pt --subset test --workers 1 --out results/p8_student.md` |
| 分离 RTF（中位数） | 0.0110（×91.3 实时） |
| 总耗时 | 51.1 min（音频总长 207.8 min，含 museval，1 进程） |

**指标：cSDR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| student | 0.64 | 4.34 | 2.45 | 1.22 | **2.17** |

**指标：ISR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| student | 2.52 | 10.54 | 3.63 | 9.44 | **6.53** |

**指标：SIR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| student | 7.76 | 6.98 | 8.05 | -1.22 | **5.39** |

**指标：SAR（中位数聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| student | 0.01 | 2.92 | 0.99 | 4.08 | **2.00** |

**指标：uSDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| student | 1.78 | 4.76 | 2.97 | 1.05 | **2.64** |

**指标：SI-SDR（均值聚合）（dB，越高越好）**

| 模型 | vocals | drums | bass | other | 平均 |
|---|---|---|---|---|---|
| student | -5.08 | 3.02 | -0.72 | -1.95 | **-1.18** |
