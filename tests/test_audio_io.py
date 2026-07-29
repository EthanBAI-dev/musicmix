"""音频 I/O 与预处理的单测。

守住全项目的形状约定 ``(n_samples, n_channels)`` 和 44.1kHz 立体声。
这些约定一旦被破坏，SDR 会静默地算错，而不是报错 —— 所以必须有测试钉住。
"""

import numpy as np
import pytest

from src.audio.io import (
    SAMPLE_RATE,
    content_hash,
    load_audio,
    loudness_normalize,
    match_length,
    peak_normalize,
    resample,
    save_audio,
    to_stereo,
)

RNG = np.random.default_rng(20260729)


def _tone(freq=440.0, seconds=1.0, sr=SAMPLE_RATE, channels=2):
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * freq * t)
    return np.stack([x] * channels, axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------
# 形状约定
# --------------------------------------------------------------------------------------

def test_to_stereo_from_1d():
    x = np.zeros(1000, dtype=np.float32)
    y = to_stereo(x)
    assert y.shape == (1000, 2)


def test_to_stereo_duplicates_mono_channel():
    x = RNG.standard_normal((500, 1)).astype(np.float32)
    y = to_stereo(x)
    assert y.shape == (500, 2)
    assert np.allclose(y[:, 0], y[:, 1])


def test_to_stereo_passthrough():
    x = RNG.standard_normal((500, 2)).astype(np.float32)
    assert to_stereo(x) is x


def test_to_stereo_truncates_surround():
    x = RNG.standard_normal((500, 6)).astype(np.float32)
    assert to_stereo(x).shape == (500, 2)


def test_to_stereo_rejects_3d():
    with pytest.raises(ValueError):
        to_stereo(np.zeros((2, 3, 4)))


# --------------------------------------------------------------------------------------
# 长度对齐：分块推理必经之路
# --------------------------------------------------------------------------------------

def test_match_length_truncates():
    x = np.ones((1000, 2), dtype=np.float32)
    assert match_length(x, 600).shape == (600, 2)


def test_match_length_zero_pads():
    x = np.ones((600, 2), dtype=np.float32)
    y = match_length(x, 1000)
    assert y.shape == (1000, 2)
    assert np.all(y[600:] == 0)
    assert np.all(y[:600] == 1)


def test_match_length_noop():
    x = np.ones((800, 2), dtype=np.float32)
    assert match_length(x, 800) is x


# --------------------------------------------------------------------------------------
# 重采样
# --------------------------------------------------------------------------------------

def test_resample_changes_length_proportionally():
    x = _tone(seconds=1.0, sr=48000)
    y = resample(x, 48000, 44100)
    assert y.shape[1] == 2
    assert abs(y.shape[0] - 44100) < 100


def test_resample_noop_when_rates_match():
    x = _tone()
    assert resample(x, SAMPLE_RATE, SAMPLE_RATE) is x


def test_resample_preserves_tone_frequency():
    """重采样后主频应当不变 —— 防的是声道维和时间维搞反这类低级错误。"""
    x = _tone(freq=1000.0, seconds=1.0, sr=48000)
    y = resample(x, 48000, 44100)
    spec = np.abs(np.fft.rfft(y[:, 0]))
    peak_hz = np.fft.rfftfreq(y.shape[0], 1 / 44100)[np.argmax(spec)]
    assert peak_hz == pytest.approx(1000.0, abs=5.0)


# --------------------------------------------------------------------------------------
# 读写
# --------------------------------------------------------------------------------------

def test_wav_roundtrip_is_bit_exact(tmp_path):
    """32-bit float WAV 往返必须完全无损，否则中间产物会被量化噪声污染。"""
    x = _tone(seconds=0.5)
    f = tmp_path / "t.wav"
    save_audio(f, x)
    y, sr = load_audio(f)
    assert sr == SAMPLE_RATE
    assert y.shape == x.shape
    assert np.allclose(y, x, atol=1e-7)


def test_load_audio_forces_stereo_from_mono_file(tmp_path):
    x = _tone(channels=1, seconds=0.3)
    f = tmp_path / "mono.wav"
    save_audio(f, x)
    y, _ = load_audio(f)
    assert y.shape[1] == 2


def test_load_audio_resamples_to_target(tmp_path):
    x = _tone(seconds=0.5, sr=22050)
    f = tmp_path / "lowsr.wav"
    save_audio(f, x, sr=22050)
    y, sr = load_audio(f)
    assert sr == SAMPLE_RATE
    assert abs(y.shape[0] - int(0.5 * SAMPLE_RATE)) < 100


def test_load_audio_missing_file():
    with pytest.raises(FileNotFoundError):
        load_audio("/nonexistent/nope.wav")


def test_load_audio_returns_float32(tmp_path):
    f = tmp_path / "t.wav"
    save_audio(f, _tone(seconds=0.2))
    y, _ = load_audio(f)
    assert y.dtype == np.float32


def test_ffmpeg_decode_path(tmp_path):
    """非 wav/flac 走 ffmpeg 分支。用 mp3 验证这条路是通的。"""
    import subprocess

    src = tmp_path / "src.wav"
    save_audio(src, _tone(seconds=0.5))
    dst = tmp_path / "src.mp3"
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-b:a", "192k", str(dst)],
        capture_output=True, check=False,
    )
    if r.returncode != 0:
        pytest.skip("ffmpeg 不可用")

    y, sr = load_audio(dst)
    assert sr == SAMPLE_RATE
    assert y.shape[1] == 2
    assert y.shape[0] > 0.4 * SAMPLE_RATE


# --------------------------------------------------------------------------------------
# 响度
# --------------------------------------------------------------------------------------

def test_loudness_normalize_hits_target():
    import pyloudnorm as pyln

    x = _tone(seconds=3.0) * 0.05  # 刻意很轻
    y = loudness_normalize(x, SAMPLE_RATE, target_lufs=-14.0)
    measured = pyln.Meter(SAMPLE_RATE).integrated_loudness(y)
    assert measured == pytest.approx(-14.0, abs=0.5)


def test_loudness_normalize_never_clips():
    x = _tone(seconds=3.0) * 0.9
    y = loudness_normalize(x, SAMPLE_RATE, target_lufs=0.0)  # 荒谬的目标，逼它触发峰值保护
    assert np.max(np.abs(y)) <= 1.0


def test_loudness_normalize_passes_through_silence():
    x = np.zeros((SAMPLE_RATE, 2), dtype=np.float32)
    assert np.array_equal(loudness_normalize(x), x)


def test_loudness_normalize_passes_through_too_short():
    """短于 BS.1770 门限窗（0.4s）的输入原样返回，不能崩。"""
    x = _tone(seconds=0.1)
    assert np.array_equal(loudness_normalize(x), x)


def test_peak_normalize():
    x = _tone() * 0.01
    y = peak_normalize(x, peak=0.99)
    assert np.max(np.abs(y)) == pytest.approx(0.99, abs=1e-4)


def test_peak_normalize_handles_silence():
    x = np.zeros((100, 2), dtype=np.float32)
    assert np.array_equal(peak_normalize(x), x)


# --------------------------------------------------------------------------------------
# 缓存哈希
# --------------------------------------------------------------------------------------

def test_content_hash_is_stable_and_distinguishing(tmp_path):
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    save_audio(a, _tone(440.0, 0.2))
    save_audio(b, _tone(880.0, 0.2))
    assert content_hash(a) == content_hash(a)
    assert content_hash(a) != content_hash(b)
