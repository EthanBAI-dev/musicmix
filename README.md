# 音乐理解与智能混音助手

> 上传一首歌 → 拆成人声/鼓/贝斯/其他四轨 → 自动分析 BPM、调性、和弦、曲式结构 → 打上风格/情绪/乐器标签 → 检索相似歌曲 → 在浏览器里重新混音。

**当前可用：上传本地歌曲 → htdemucs 四轨分离 → BPM/调性/和弦/曲式分析 → 浏览器混音与 WAV 导出。**

## 快速开始

macOS 需要先安装 FFmpeg（`brew install ffmpeg`）。Windows/Linux 请用系统包管理器安装 FFmpeg。

```bash
git clone https://github.com/EthanBAI-dev/musicmix.git
cd musicmix
conda create -n music-mix python=3.11 -y
conda activate music-mix
python -m pip install -e ".[sep]"
python -m scripts.make_demo_stems
python -m scripts.serve
```

打开 <http://localhost:8123/web/index.html>，点击“上传并分离”。首次分离会从
HTDemucs 官方 Hugging Face 仓库自动下载约 **84 MB** 权重，之后使用本机缓存。

> 权重没有复制进本仓库：它已接近 GitHub 普通文件的 100 MB 上限，
> 而且 Demucs 本身能校验并缓存官方权重。这样 clone 更快，模型来源也更清楚。

开发与评测环境：

```bash
python -m pip install -e ".[eval,torch,sep,dev]"
python -m scripts.check_env
python -m pytest tests/ -q
```

## 本地使用

```bash
python -m scripts.serve
```

然后打开 <http://localhost:8123/web/index.html>。

> 用 `scripts.serve` 而不是 `python -m http.server`：后者只发 `Last-Modified`，
> 浏览器会缓存 css/js，改完刷新看不到变化，且症状极具迷惑性（详见 DEVLOG 2026-07-30）。

**页面能做什么**
- 四轨播放：每轨独立 音量 / 静音 M / 独奏 S / 声像 / 三段 EQ，所有增益变化都做平滑，无爆音
- 时间轴：曲式段落色块、和弦标签、小节线与拍点网格，点击定位
- 频谱分析（对数频率轴）、逐轨电平表、波形图
- 导出 WAV（`OfflineAudioContext` 离线渲染，20 秒曲子约 130ms）
- 快捷键：空格播放/暂停、←/→ 前后 2 秒、Home 回到开头
- 上传一首完整歌曲：后台自动分成 `vocals` / `drums` / `bass` / `other`
- 载入已经分离的音轨：文件名含上述四个名称即可
- **P1 失败案例听审**：下拉选案例，一键在「分离结果 / 真值」间 A/B（快捷键 `Tab`），
  切换时保持播放位置不变；每轨标注 cSDR 并区分「真实失败」与「指标假象」

演示音频由 `scripts/make_demo_stems.py` 合成（20 秒 / 96 BPM / A 小调 / Am-F-C-G）。
它的 BPM、拍点、和弦、曲式**全部是已知真值**，所以将来会作为 P3 拍点/和弦/结构模型的第一个测试样例。

---

## 文档导航

| 文档 | 看什么 |
|---|---|
| [00-总体大纲](docs/00-总体大纲.md) | **从这里开始。** 架构图、技术栈、P0~P8 每个阶段做什么、「我做了什么」清单、风险与降级方案 |
| [01-术语表](docs/01-术语表.md) | 项目里出现的每个专业名词的解释（SDR、mAP、Macro-F1、Band-Split、RoPE、AudioWorklet…） |
| [02-方案调研与选型](docs/02-方案调研与选型.md) | 开源方案横向对比、选型理由、参考链接、**「不做什么」清单** |
| [03-评测协议](docs/03-评测协议.md) | 每个指标怎么算、结果表模板、显著性检验、失败案例规范 |
| [04-里程碑与排期](docs/04-里程碑与排期.md) | 9 周计划，逐项勾选清单 |
| [DEVLOG](docs/DEVLOG.md) | **开发日志**，每一步的详细记录 |

---

## 这个项目想证明什么

不是"能把开源模型串起来"，而是：

| 能力 | 证据 |
|---|---|
| 会做可信的评测 | 复现论文数字（±0.3 dB）、平凡基线 + oracle 上界、配对 bootstrap 显著性检验 |
| 会做真正的改进 | 逐项累加的消融表，而不是"换了个模型效果更好" |
| 有研究思维 | 提出可证伪假设（Stem-aware 的收益主要来自乐器类标签）并用分类别指标验证 |
| 懂工程权衡 | SDR–RTF 帕累托曲线、HNSW 召回/速度权衡、端到端延迟明细 |
| 诚实 | 失败案例专章、代理真值的局限声明、负结果照样写 |

---

## 核心指标（待填）

| 任务 | 指标 | Baseline | Ours |
|---|---|---|---|
| 音源分离 | cSDR (MUSDB18-HQ test 50 首, 4 轨平均) | **htdemucs 8.80 dB**（官方 9.00，差 0.20）| 待 P2 |
| 音乐标签 | mAP / Macro-F1 (MTG-Jamendo) | — | — |
| 相似检索 | HitRate@10 | — | — |
| 推理性能 | 端到端 RTF | — | — |

MUSDB18-HQ test（50 首）上的完整锚点（cSDR，museval 中位数聚合，[P1 报告](results/P1_分离baseline.md)）：

| 锚点 | cSDR 平均 | 作用 |
|---|---|---|
| `silence` 输出全零 | uSDR 恰为 **0.00 dB** | 解析已知值，验证实现正确 |
| `trivial` 混音当每一轨 | -5.34 dB | **下界** |
| `htdemucs` | **8.80 dB** | 复现官方 9.00 dB |
| `oracle` IRM 理想掩码 | 8.95 dB | **掩码类方法的上界** |

> IRM oracle 只是**掩码方法**的上界，不是通用上界 —— Demucs 直接生成波形、
> 能修正相位，实测在 drums/bass 上已与它统计上不可区分（配对 bootstrap p=0.88 / 0.18）。
> M0 时我预期 oracle 在 12~15 dB，那一档其实对应多通道维纳滤波 oracle，预期是错的。

---

## 数据与许可

- **MUSDB18-HQ** — 150 首，非商业研究用途。音频不入库。
- **MTG-Jamendo** — 55,525 首 / 183 标签。元数据 CC BY-NC-SA 4.0，**仅限非商业研究与学术使用；商业使用需 Jamendo S.A. 书面授权**。

本项目为个人学习与研究作品，不作商业用途。

用户上传的原始音频只在本机临时存放，任务结束后删除；分离结果保存在
`web/mine/` 供浏览器播放，该目录已被 Git 忽略，不会被上传。服务默认仅监听
`127.0.0.1`，不向局域网或公网开放。

---

## 环境

```
Apple M2 Max / 64GB · conda env `music-mix` (Python 3.11) · PyTorch 2.13 (MPS) · demucs 4.1.0 · Node 24
```
