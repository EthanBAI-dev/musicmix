"""合成"玩具歌"，用于在没有真实数据集时验证评测框架。

.. warning::
   **踩坑记录（2026-07-29，值得永远记住）**

   第一版合成器用的是**纯正弦** + **纯延迟做立体声**（``np.roll``）。
   结果 cSDR / ISR 看着正常，但 **SIR ≈ -114 dB、SAR 恒为 0.00 dB**，全是垃圾值。

   原因不在评测代码，在数据本身：BSS Eval v4 是把估计投影到"全部参考经 512 阶 FIR
   滤波"张成的子空间上再做分解的。而

   - **一个纯正弦可以被 2 阶 FIR 变成同频率的任意相位/幅度** —— 子空间严重退化
   - **纯延迟正是一个 FIR 抽头** —— 左右声道线性相关，进一步退化

   投影的最小二乘问题于是病态，SIR/SAR 失去意义。

   所以这一版：① 每个声部都含**带限噪声**（宽带、随机，FIR 无法预测）；
   ② 音高**随时间变化**（破坏"同频正弦可互相转换"的退化）；
   ③ 立体声靠**独立的去相关噪声**，不用延迟。

   教训推广到真实工作：**评测指标异常时，先怀疑数据是否落在指标的适用假设之外，
   再怀疑指标实现。**
"""

from __future__ import annotations

import numpy as np

SR = 44100
STEMS = ("vocals", "drums", "bass", "other")


def _bandpass_noise(rng: np.random.Generator, n: int, lo: float, hi: float, sr: int = SR) -> np.ndarray:
    """频域整形的带限噪声。宽带且随机 —— FIR 滤波器无法预测，正是我们需要的。"""
    spec = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    freqs = np.fft.rfftfreq(n, 1 / sr)
    # 平滑的带通包络，避免砖墙滤波带来的振铃
    band = np.exp(-0.5 * ((np.log(np.maximum(freqs, 1e-3)) - np.log(np.sqrt(lo * hi)))
                          / (0.5 * np.log(hi / lo))) ** 2)
    band[freqs < lo * 0.3] = 0.0
    band[freqs > hi * 3.0] = 0.0
    y = np.fft.irfft(spec * band, n=n)
    peak = np.max(np.abs(y))
    return y / peak if peak > 0 else y


def _wandering_tone(rng: np.random.Generator, n: int, f_center: float, sr: int = SR) -> np.ndarray:
    """音高随时间缓慢游走的谐波音。

    频率变化是关键：固定频率的正弦对 512 阶 FIR 来说是平凡的，会让 BSS Eval 退化。
    """
    t = np.arange(n) / sr
    drift = 1.0 + 0.06 * np.sin(2 * np.pi * rng.uniform(0.2, 0.6) * t + rng.uniform(0, 6.28))
    vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * rng.uniform(4.5, 6.5) * t)
    phase = 2 * np.pi * np.cumsum(f_center * drift * vibrato) / sr
    return np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)


def _to_stereo(rng: np.random.Generator, mono: np.ndarray, width: float = 0.25) -> np.ndarray:
    """用**独立噪声**做去相关的立体声形象。

    绝不用 ``np.roll`` 那种纯延迟 —— 纯延迟就是一个 FIR 抽头，会让左右声道线性相关。
    """
    n = mono.shape[0]
    decorr_l = _bandpass_noise(rng, n, 200, 8000) * width * 0.15
    decorr_r = _bandpass_noise(rng, n, 200, 8000) * width * 0.15
    left = mono * (1.0 - width * 0.3) + decorr_l
    right = mono * (1.0 + width * 0.3) + decorr_r
    return np.stack([left, right], axis=1).astype(np.float32)


def make_synthetic_track(
    seed: int, seconds: float = 6.0, sr: int = SR
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """造一首玩具歌：四个频谱重心不同的声部 + 它们的和。

    声部之间**刻意保留一些频谱重叠**，这样 IRM oracle 也到不了 +∞，
    数字更接近真实情况。

    Returns:
        ``(references {stem: (n, 2)}, mixture (n, 2))``
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr

    # bass：低频游走音 + 低频噪声底
    bass_m = 0.40 * _wandering_tone(rng, n, rng.uniform(45, 75), sr) + 0.06 * _bandpass_noise(rng, n, 30, 250)

    # drums：宽带噪声 + 周期性冲击包络
    bpm = rng.uniform(90, 140)
    hit_env = np.abs(np.sin(np.pi * bpm / 60 * t)) ** 8
    drums_m = _bandpass_noise(rng, n, 60, 16000) * (0.08 + 0.5 * hit_env)

    # vocals：中频游走音 + 少量气声噪声
    voc_env = 0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * t) ** 2
    vocals_m = (0.30 * _wandering_tone(rng, n, rng.uniform(180, 320), sr) * voc_env
                + 0.04 * _bandpass_noise(rng, n, 1500, 9000) * voc_env)

    # other：中高频和声 + 弦乐质感噪声
    other_m = (0.18 * _wandering_tone(rng, n, rng.uniform(500, 1100), sr)
               + 0.10 * _bandpass_noise(rng, n, 800, 12000))

    refs = {
        "vocals": _to_stereo(rng, vocals_m, width=0.15),
        "drums": _to_stereo(rng, drums_m, width=0.35),
        "bass": _to_stereo(rng, bass_m, width=0.08),
        "other": _to_stereo(rng, other_m, width=0.45),
    }
    mixture = sum(refs.values()).astype(np.float32)

    # 防削顶：整体缩放，不改变各声部之间的比例（SDR 对整体缩放不敏感，但削顶会）
    peak = np.max(np.abs(mixture))
    if peak > 0.95:
        scale = 0.95 / peak
        refs = {k: (v * scale).astype(np.float32) for k, v in refs.items()}
        mixture = (mixture * scale).astype(np.float32)

    return refs, mixture


class SyntheticTrack:
    """鸭子类型地模仿 :class:`src.datasets.musdb.Track`，让下游代码一视同仁。"""

    def __init__(self, seed: int, seconds: float = 6.0):
        self.name = f"synthetic-{seed:02d}"
        self.subset = "synthetic"
        self._refs, self._mix = make_synthetic_track(seed, seconds)
        self.duration_seconds = seconds

    def references(self) -> dict[str, np.ndarray]:
        return self._refs

    def mixture(self) -> np.ndarray:
        return self._mix

    def audio(self, stem: str) -> np.ndarray:
        return self._mix if stem == "mixture" else self._refs[stem]
