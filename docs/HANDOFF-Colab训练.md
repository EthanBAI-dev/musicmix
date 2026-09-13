# Colab 训练交接

更新时间：2026-09-13

## 当前状态

- Google Drive 根目录：`/content/drive/MyDrive/Audio AI/MusicMixer`
- Colab 已切换为 Python 3.12，GPU 为 Tesla T4（约 14.6 GiB 显存）。
- MERT-v1-95M 固定 revision：`12af15fef9d0ac838c3f475bfbbf26d2060dd4f5`
- 冒烟实验目录：`runs/smoke-last-seed0`
- 缺失的 `meta/splits/split-0/autotagging_top50tags-{train,validation,test}.tsv` 已给出补齐方法。
- 训练尚未开始；数据预检在 `08/808.mp3` 处停止，说明脚本既没找到解开的音频，也没找到 `audio_tar/raw_30s_audio-08.tar`。
- 已有截图确认 `audio_tar` 至少包含第 10–18 块，不能据此认定 00–99 全部齐全。

## 恢复步骤

先在 Colab 运行只读清单检查：

```python
from pathlib import Path

BASE = Path('/content/drive/MyDrive/Audio AI/MusicMixer')
folder = BASE / 'audio_tar'
missing = [
    i for i in range(100)
    if not (folder / f'raw_30s_audio-{i:02d}.tar').is_file()
]
print('缺少的音频包编号：', missing)
```

若缺失列表正好是 `0..9`，只下载这些块：

```python
import subprocess, sys

subprocess.run([
    sys.executable, '-m', 'scripts.cloud_download',
    '--root', str(BASE), '--start', '0', '--stop', '10',
], cwd=str(BASE / 'project'), check=True)
```

若缺失列表不同，按连续区间分别运行 `--start/--stop`。下载器会保留已验证块，不应删除已有 tar。补齐后保持 `SMOKE=True`、`RESUME=False`，重新运行实时日志训练单元。

## 成功标准

冒烟训练日志应出现 epoch 进度和 `val_mAP`，并生成：

- `runs/smoke-last-seed0/best.pt`
- `runs/smoke-last-seed0/last.pt`
- `runs/smoke-last-seed0/history.jsonl`
- `runs/smoke-last-seed0/validation_predictions.npz`

冒烟结果仅验证工程管线，不能写入正式模型成绩。成功后按 `colab/README.md` 运行全量 frozen 对照和 last-2 微调。

## 已交付代码

- `colab/train_mert.ipynb`：Colab 入口
- `scripts/cloud_download.py`：全量 tar 续传与完整性检查
- `scripts/cloud_train.py`：冻结/局部/全量微调、断点恢复和独立测试
- `docs/05-架构审查与云端训练方案.md`：架构、模型、商业化和实验审查
- `tests/test_cloud_training.py`、`tests/test_cloud_mert_compat.py`：训练与官方 MERT 代码兼容性测试

## 验证记录

- 55 项相关本地测试通过，2 项联网 MERT 兼容性测试在隔离 Transformers 4.44.2 环境通过。
- 尚未在 Colab 上完成完整预训练权重的冒烟训练，因此不宣称显存、速度或指标已经验证。
