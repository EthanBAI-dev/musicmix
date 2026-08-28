"""音频 → mel 频谱特征，并缓存到磁盘。

L0（mel + 小 CNN 从头训）用的输入。为什么要缓存：
mp3 解码 + mel 变换在 CPU 上大约 0.3 s/首，5500 首就是半小时。
训练要跑几十个 epoch，每个 epoch 重算一遍完全不可接受。
缓存成 float16 的 ``.npy``（**每首约 470 KB**），之后每个 epoch 只是读盘。

约定（改之前想清楚，缓存和模型都依赖它）：

- **16 kHz 单声道**。音乐标签任务的通行设置，MusiCNN / 多数 baseline 都是 16k；
  高频对风格/情绪判别贡献很小，降采样能让 mel 帧数和显存都降下来。
  （注意这和分离流水线的 44.1kHz 立体声是**两套独立约定**，别混用。）
- **每首取中间 30 秒**。MTG-Jamendo 的曲目平均 244 秒，全曲会让长曲目在
  训练里超额加权；取固定中段也让缓存大小可预测。
- mel 128 bin / n_fft 512 / hop 256 → 30 s ≈ **1876 帧**
- 存 **对数 mel**（``log(1 + mel)``），不存线性值 —— 线性 mel 的动态范围跨好几个数量级，
  float16 存不下且不利于训练。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 16000
N_FFT = 512
HOP = 256
N_MELS = 128
CLIP_SECONDS = 30.0
N_FRAMES = int(CLIP_SECONDS * SR / HOP) + 1        # 1876


@dataclass(frozen=True)
class MelConfig:
    sr: int = SR
    n_fft: int = N_FFT
    hop: int = HOP
    n_mels: int = N_MELS
    clip_seconds: float = CLIP_SECONDS

    @property
    def n_frames(self) -> int:
        return int(self.clip_seconds * self.sr / self.hop) + 1

    def tag(self) -> str:
        """缓存目录名。参数一变就换目录，避免读到用旧参数算的特征。"""
        return f"mel{self.n_mels}_sr{self.sr}_hop{self.hop}_{self.clip_seconds:.0f}s"


def compute_mel(path: Path, cfg: MelConfig = MelConfig()) -> np.ndarray:
    """一首歌 → ``(n_mels, n_frames) float16`` 的对数 mel。

    取中间 ``clip_seconds`` 秒；不足则整首取出后补零到固定长度
    （补零而不是循环填充：循环会造出原曲没有的重复结构）。
    """
    import librosa

    duration = librosa.get_duration(path=str(path))
    offset = max(0.0, (duration - cfg.clip_seconds) / 2)
    y, _ = librosa.load(str(path), sr=cfg.sr, mono=True,
                        offset=offset, duration=cfg.clip_seconds)

    need = int(cfg.clip_seconds * cfg.sr)
    if len(y) < need:
        y = np.pad(y, (0, need - len(y)))

    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sr, n_fft=cfg.n_fft, hop_length=cfg.hop, n_mels=cfg.n_mels)
    logmel = np.log1p(mel).astype(np.float16)

    # 长度可能因取整差 1 帧，统一钉死，否则拼 batch 时会炸
    if logmel.shape[1] != cfg.n_frames:
        logmel = (logmel[:, : cfg.n_frames] if logmel.shape[1] > cfg.n_frames
                  else np.pad(logmel, ((0, 0), (0, cfg.n_frames - logmel.shape[1]))))
    return logmel


def cache_path(track_path: str, root: Path, cfg: MelConfig = MelConfig()) -> Path:
    """``"14/214.mp3"`` → ``<root>/features/mel128_.../14/214.npy``"""
    return root / "features" / cfg.tag() / track_path.replace(".mp3", ".npy")


def load_or_compute(track_path: str, root: Path, cfg: MelConfig = MelConfig(),
                    audio_dir: str = "audio") -> np.ndarray:
    dst = cache_path(track_path, root, cfg)
    if dst.exists():
        return np.load(dst)
    mel = compute_mel(root / audio_dir / track_path, cfg)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, mel)
    return mel


def normalize_stats(mels: list[np.ndarray]) -> tuple[float, float]:
    """训练集上的全局均值/标准差。

    **必须只用训练集算**，然后原样套到验证/测试集 —— 用全体数据算就是信息泄漏。
    """
    cat = np.concatenate([m.astype(np.float32).reshape(-1) for m in mels])
    return float(cat.mean()), float(cat.std() + 1e-8)
