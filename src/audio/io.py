"""音频 I/O 与预处理。

全项目统一约定（改动这里之前先想清楚，所有评测都依赖它）：

- 波形数组形状恒为 ``(n_samples, n_channels)``，float32。
  这是 soundfile / museval / musdb 的约定；torchaudio 用的是 ``(n_channels, n_samples)``，
  所以每次和 torch 交互都要显式转置，别靠记忆。
- 采样率恒为 44100 Hz，**立体声**。
  MUSDB18-HQ 就是这个规格，转单声道会让 SDR 无法与论文比较。
- 单声道文件会被复制成两条相同的声道，而不是保持单声道。
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 44100
N_CHANNELS = 2
TARGET_LUFS = -14.0

# soundfile 能直接读的格式；其余（mp3/m4a/...）走 ffmpeg 解码
_NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".w64"}


def to_stereo(x: np.ndarray) -> np.ndarray:
    """把任意声道数的波形规整为立体声 ``(n, 2)``。

    - ``(n,)`` 或 ``(n, 1)``：复制成两条相同声道
    - ``(n, 2)``：原样返回
    - ``(n, >2)``：只取前两条（MUSDB 不会出现，但真实上传文件会）
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"期望 1D 或 2D 波形，得到 shape={x.shape}")
    n_ch = x.shape[1]
    if n_ch == 1:
        return np.repeat(x, 2, axis=1)
    if n_ch == 2:
        return x
    return x[:, :2]


def resample(x: np.ndarray, sr_in: int, sr_out: int = SAMPLE_RATE) -> np.ndarray:
    """重采样。用 soxr（librosa 后端）保证质量，避免廉价线性插值引入的混叠。"""
    if sr_in == sr_out:
        return x
    import librosa

    # librosa 按 (channels, samples) 处理多声道，这里临时转置
    y = librosa.resample(x.T, orig_sr=sr_in, target_sr=sr_out, res_type="soxr_hq")
    return np.ascontiguousarray(y.T)


def _decode_with_ffmpeg(path: Path, sr: int) -> np.ndarray:
    """用 ffmpeg 解码任意格式为 float32 立体声 PCM。

    直接读 stdout 的裸 PCM，不落临时文件。ffmpeg 已在环境里（v8.1）。
    """
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ar", str(sr), "-ac", str(N_CHANNELS),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 解码失败：{path}\n{proc.stderr.decode(errors='replace')}")
    audio = np.frombuffer(proc.stdout, dtype="<f4")
    return audio.reshape(-1, N_CHANNELS).astype(np.float32)


def load_audio(
    path: str | Path,
    sr: int = SAMPLE_RATE,
    stereo: bool = True,
    normalize_loudness: bool = False,
) -> tuple[np.ndarray, int]:
    """读一个音频文件，返回 ``(waveform (n, ch) float32, sr)``。

    Args:
        path: 音频路径。
        sr: 目标采样率，默认 44100。
        stereo: 是否强制立体声。**评测流程必须保持 True。**
        normalize_loudness: 是否做 -14 LUFS 响度归一化。
            注意：**分离评测时必须为 False** —— 归一化会改变波形幅度，
            而 SDR 是幅度敏感的（SI-SDR 才不敏感）。只在用户上传的推理路径上开。

    Returns:
        (waveform, sr)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"音频不存在：{path}")

    if path.suffix.lower() in _NATIVE_SUFFIXES:
        import soundfile as sf

        x, sr_in = sf.read(str(path), dtype="float32", always_2d=True)
        x = resample(x, sr_in, sr)
    else:
        x = _decode_with_ffmpeg(path, sr)

    if stereo:
        x = to_stereo(x)
    x = np.ascontiguousarray(x, dtype=np.float32)

    if normalize_loudness:
        x = loudness_normalize(x, sr)
    return x, sr


def save_audio(path: str | Path, x: np.ndarray, sr: int = SAMPLE_RATE, subtype: str = "FLOAT") -> None:
    """写音频。默认 32-bit float WAV，避免中间产物被 16-bit 量化噪声污染。"""
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.asarray(x, dtype=np.float32), sr, subtype=subtype)


def loudness_normalize(
    x: np.ndarray, sr: int = SAMPLE_RATE, target_lufs: float = TARGET_LUFS
) -> np.ndarray:
    """ITU-R BS.1770 响度归一化到 target_lufs。

    为什么要做：不同歌的音量差十几 dB，直接喂模型会让"响"变成一个假特征。
    为什么评测时不做：见 :func:`load_audio` 的 docstring。

    静音或极短（<0.4s，短于 BS.1770 的门限窗）的输入原样返回。
    """
    import pyloudnorm as pyln

    if x.size == 0 or not np.any(x):
        return x
    if x.shape[0] < int(0.4 * sr):
        return x

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(x)
    if not np.isfinite(loudness):  # 全静音时返回 -inf
        return x
    y = pyln.normalize.loudness(x, loudness, target_lufs)

    # 归一化可能推过 0 dBFS，做一次峰值保护（保留 1 dB 余量）
    peak = np.max(np.abs(y))
    if peak > 0.891:  # 10^(-1/20)
        y = y * (0.891 / peak)
    return np.ascontiguousarray(y, dtype=np.float32)


def peak_normalize(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """峰值归一化。只在需要快速对齐音量做主观试听时用，不用于评测。"""
    m = np.max(np.abs(x))
    if m == 0:
        return x
    return np.ascontiguousarray(x * (peak / m), dtype=np.float32)


def match_length(x: np.ndarray, n: int) -> np.ndarray:
    """把波形补零或截断到 n 个采样点。

    分离模型的分块推理经常让输出比输入长几百个点，评测前必须对齐，
    否则 museval 会直接报错或静默错位。
    """
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    pad = np.zeros((n - x.shape[0], x.shape[1]), dtype=x.dtype)
    return np.concatenate([x, pad], axis=0)


def content_hash(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """音频文件内容哈希，用于 P6 的结果缓存（同一首歌不重复计算）。"""
    import hashlib

    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
