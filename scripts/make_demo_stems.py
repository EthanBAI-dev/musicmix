"""生成前端演示用的四轨音频 + 分析真值。

    python -m scripts.make_demo_stems

为什么需要它：P7 混音台要能演示，但 P1/P2 的真实分离结果还没有。
这里用加法合成造一段 20 秒的小曲子，直接产出**四个独立声部**，
前端可以立刻当作"已分离好的 stem"来用。

额外的好处：**这段曲子的 BPM、调性、和弦、拍点、段落全部是已知真值**，
所以 ``analysis.json`` 不是猜的。P3 接入 beat_this / all-in-one 之后，
可以拿它当第一个"必须答对"的测试样例 —— 连自己生成的规整曲子都跟不准，
就别谈真实音乐了。

输出到 ``web/demo/``：mixture + 4 stems（mp3）+ analysis.json

.. note::
   **各声部是独立做 mp3 编码的，所以浏览器里 Σstems ≠ mixture。**
   实测四轨相加与 ``mixture.mp3`` 的误差能量约为信号的 -20 dB，
   最大偏差出现在鼓过门这类瞬态处（mp3 的前回声 pre-echo）。时间对齐没有问题
   （扫描 ±1152 采样，最佳偏移就是 0），纯粹是有损编码各自量化的结果。

   在 float 域里这条不变量是精确成立的（本脚本自检误差 ~1e-7）。
   P1 输出真实分离结果时要记住这一点：**不要在有损编码之后去验证
   「四轨之和等于混音」，那样验的是编解码器，不是分离模型。**
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from src.audio.io import save_audio

SR = 44100
BPM = 96.0
BEATS_PER_BAR = 4
N_BARS = 8
OUT_DIR = Path("web/demo")

SEC_PER_BEAT = 60.0 / BPM
SEC_PER_BAR = SEC_PER_BEAT * BEATS_PER_BAR
TOTAL_SEC = SEC_PER_BAR * N_BARS

# A 小调，一小节一个和弦
PROGRESSION = ["Am", "F", "C", "G", "Am", "F", "C", "G"]
CHORD_NOTES = {          # 三和弦（Hz）
    "Am": (220.00, 261.63, 329.63),
    "F": (174.61, 220.00, 261.63),
    "C": (261.63, 329.63, 392.00),
    "G": (196.00, 246.94, 293.66),
}
BASS_ROOT = {"Am": 110.00, "F": 87.31, "C": 130.81, "G": 98.00}

# 曲式：前 2 小节 intro，中间 4 小节 verse，最后 2 小节 chorus
SEGMENTS = [
    {"start": 0.0, "end": 2 * SEC_PER_BAR, "label": "intro"},
    {"start": 2 * SEC_PER_BAR, "end": 6 * SEC_PER_BAR, "label": "verse"},
    {"start": 6 * SEC_PER_BAR, "end": 8 * SEC_PER_BAR, "label": "chorus"},
]

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------------------
# 基础合成器
# --------------------------------------------------------------------------------------

def _t(n: int) -> np.ndarray:
    return np.arange(n) / SR


def adsr(n: int, a=0.01, d=0.1, s=0.7, r=0.2) -> np.ndarray:
    """ADSR 包络。attack/decay/release 单位是秒，sustain 是电平。"""
    env = np.zeros(n)
    na, nd = int(a * SR), int(d * SR)
    nr = min(int(r * SR), max(n - na - nd, 0))
    ns = max(n - na - nd - nr, 0)
    i = 0
    if na:
        env[i:i + na] = np.linspace(0, 1, na); i += na
    if nd:
        env[i:i + nd] = np.linspace(1, s, nd); i += nd
    if ns:
        env[i:i + ns] = s; i += ns
    if nr:
        env[i:i + nr] = np.linspace(s, 0, nr)
    return env


def saw(freq: float, n: int, n_harm: int = 12) -> np.ndarray:
    """加法合成的锯齿波。限制谐波数以避免混叠。"""
    t = _t(n)
    y = np.zeros(n)
    for k in range(1, n_harm + 1):
        if freq * k > SR / 2.2:
            break
        y += np.sin(2 * np.pi * freq * k * t) / k
    return y * (2 / np.pi)


def sine(freq: float, n: int, vibrato: float = 0.0) -> np.ndarray:
    t = _t(n)
    f = freq * (1 + vibrato * np.sin(2 * np.pi * 5.5 * t)) if vibrato else freq
    return np.sin(2 * np.pi * np.cumsum(np.full(n, 1.0) * f) / SR)


def place(buf: np.ndarray, sig: np.ndarray, start_sec: float) -> None:
    """把 sig 叠加到 buf 的 start_sec 处（越界自动截断）。"""
    i = int(start_sec * SR)
    j = min(i + len(sig), len(buf))
    if i < len(buf):
        buf[i:j] += sig[: j - i]


# --------------------------------------------------------------------------------------
# 四个声部
# --------------------------------------------------------------------------------------

def make_bass(n: int) -> np.ndarray:
    """贝斯：每小节根音，八分音符律动。"""
    out = np.zeros(n)
    for bar, chord in enumerate(PROGRESSION):
        root = BASS_ROOT[chord]
        for eighth in range(8):
            start = bar * SEC_PER_BAR + eighth * SEC_PER_BEAT / 2
            dur = SEC_PER_BEAT / 2 * 0.85
            m = int(dur * SR)
            # 第 1、5 个八分重一点，做出律动
            amp = 0.62 if eighth in (0, 4) else 0.38
            note = saw(root, m, n_harm=8) * adsr(m, a=0.005, d=0.06, s=0.55, r=0.08) * amp
            place(out, note, start)
    return out


def make_drums(n: int) -> np.ndarray:
    """鼓：底鼓在 1/3 拍，军鼓在 2/4 拍，踩镲走八分。"""
    out = np.zeros(n)

    def kick(dur=0.28):
        m = int(dur * SR)
        t = _t(m)
        # 频率从 120Hz 快速扫到 45Hz —— 底鼓的经典做法
        f = 45 + 75 * np.exp(-t * 28)
        return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 9) * 0.62

    def snare(dur=0.22):
        m = int(dur * SR)
        t = _t(m)
        noise = rng.standard_normal(m)
        # 简易高通：一阶差分，把噪声推向高频
        noise = np.diff(noise, prepend=0.0)
        tone = np.sin(2 * np.pi * 190 * t) * 0.35
        return (noise * 0.5 + tone) * np.exp(-t * 22) * 0.42

    def hihat(dur=0.07, amp=0.22):
        m = int(dur * SR)
        t = _t(m)
        noise = rng.standard_normal(m)
        noise = np.diff(noise, prepend=0.0)
        noise = np.diff(noise, prepend=0.0)  # 两次差分 → 更亮
        return noise * np.exp(-t * 60) * amp

    for bar in range(N_BARS):
        bar_t = bar * SEC_PER_BAR
        for beat in range(BEATS_PER_BAR):
            bt = bar_t + beat * SEC_PER_BEAT
            if beat in (0, 2):
                place(out, kick(), bt)
            if beat in (1, 3):
                place(out, snare(), bt)
            place(out, hihat(amp=0.26 if beat == 0 else 0.16), bt)
            place(out, hihat(amp=0.11), bt + SEC_PER_BEAT / 2)
        # 每 4 小节末尾加一个小过门，让结构听得出来
        if bar % 4 == 3:
            for i in range(4):
                place(out, snare(0.12), bar_t + 3 * SEC_PER_BEAT + i * SEC_PER_BEAT / 4)
    return out


def make_other(n: int) -> np.ndarray:
    """其他：和弦垫（pad），慢起音，撑住和声。"""
    out = np.zeros(n)
    for bar, chord in enumerate(PROGRESSION):
        m = int(SEC_PER_BAR * 0.98 * SR)
        env = adsr(m, a=0.25, d=0.3, s=0.6, r=0.5)
        layer = np.zeros(m)
        for f in CHORD_NOTES[chord]:
            # 两个略微失谐的锯齿叠在一起 → 厚度
            layer += saw(f, m, n_harm=10) * 0.5
            layer += saw(f * 1.004, m, n_harm=10) * 0.35
        place(out, layer * env * 0.30, bar * SEC_PER_BAR)
    return out


def make_vocals(n: int) -> np.ndarray:
    """主旋律：A 小调五声音阶，正弦 + 谐波 + 少量气声。

    intro 段落刻意留白（不出声），这样"段落结构"在混音台上看得见也听得出。
    """
    out = np.zeros(n)
    A4, C5, D5, E5, G5 = 440.00, 523.25, 587.33, 659.25, 783.99
    # (相对起始拍, 时值拍, 频率)；从第 2 小节（verse）开始
    melody = [
        (0, 1.5, A4), (1.5, 0.5, C5), (2, 1, D5), (3, 1, C5),
        (4, 1.5, A4), (5.5, 0.5, G5 / 2), (6, 2, A4),
        (8, 1, C5), (9, 1, D5), (10, 1.5, E5), (11.5, 0.5, D5),
        (12, 2, C5), (14, 2, A4),
        (16, 1, E5), (17, 1, G5), (18, 1.5, E5), (19.5, 0.5, D5),
        (20, 2, C5), (22, 2, A4),
    ]
    offset = 2 * SEC_PER_BAR
    for start_beat, dur_beat, freq in melody:
        start = offset + start_beat * SEC_PER_BEAT
        m = int(dur_beat * SEC_PER_BEAT * 0.92 * SR)
        if m <= 0:
            continue
        env = adsr(m, a=0.04, d=0.12, s=0.75, r=0.18)
        tone = sine(freq, m, vibrato=0.008)
        tone += 0.28 * sine(freq * 2, m) + 0.12 * sine(freq * 3, m)
        breath = rng.standard_normal(m)
        breath = np.diff(breath, prepend=0.0) * 0.03
        place(out, (tone + breath) * env * 0.40, start)
    return out


def to_stereo(mono: np.ndarray, pan: float = 0.0, width: float = 0.0) -> np.ndarray:
    """等功率声像 + 可选的去相关加宽。pan ∈ [-1, 1]。"""
    angle = (pan + 1) * np.pi / 4
    left, right = np.cos(angle), np.sin(angle)
    l_ch, r_ch = mono * left, mono * right
    if width > 0:
        n = len(mono)
        d = rng.standard_normal(n) * width * 0.012
        l_ch = l_ch + d
        r_ch = r_ch - d
    return np.stack([l_ch, r_ch], axis=1).astype(np.float32)


# --------------------------------------------------------------------------------------
# 分析真值
# --------------------------------------------------------------------------------------

def build_analysis() -> dict:
    """这段曲子的 BPM / 拍点 / 和弦 / 段落 —— 全部是**已知真值**，不是估计。

    P3 接入 beat_this / all-in-one 之后，拿它当第一个必须答对的样例。
    """
    beats, downbeats = [], []
    for bar in range(N_BARS):
        for beat in range(BEATS_PER_BAR):
            t = round(bar * SEC_PER_BAR + beat * SEC_PER_BEAT, 6)
            beats.append(t)
            if beat == 0:
                downbeats.append(t)
    chords = [
        {"start": round(i * SEC_PER_BAR, 6), "end": round((i + 1) * SEC_PER_BAR, 6), "label": c}
        for i, c in enumerate(PROGRESSION)
    ]
    return {
        "source": "synthetic (scripts/make_demo_stems.py)",
        "ground_truth": True,
        "duration": round(TOTAL_SEC, 6),
        "sample_rate": SR,
        "bpm": BPM,
        "time_signature": "4/4",
        "key": "A minor",
        "beats": [round(b, 6) for b in beats],
        "downbeats": [round(b, 6) for b in downbeats],
        "chords": chords,
        "segments": [
            {"start": round(s["start"], 6), "end": round(s["end"], 6), "label": s["label"]}
            for s in SEGMENTS
        ],
        "stems": ["vocals", "drums", "bass", "other"],
    }


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def encode_mp3(wav: Path, mp3: Path, bitrate: str = "160k") -> bool:
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(wav), "-b:a", bitrate, str(mp3)],
        capture_output=True, check=False,
    )
    return r.returncode == 0


def main() -> int:
    n = int(TOTAL_SEC * SR)
    print(f"生成 {TOTAL_SEC:.1f}s / {BPM:.0f} BPM / {PROGRESSION[0]}-{PROGRESSION[1]}-"
          f"{PROGRESSION[2]}-{PROGRESSION[3]} 循环 …")

    stems = {
        "vocals": to_stereo(make_vocals(n), pan=0.0, width=0.3),
        "drums": to_stereo(make_drums(n), pan=0.0, width=0.6),
        "bass": to_stereo(make_bass(n), pan=0.0, width=0.05),
        "other": to_stereo(make_other(n), pan=0.0, width=0.9),
    }
    mixture = sum(stems.values()).astype(np.float32)

    # 统一缩放防削顶。**必须四轨等比例缩放**，否则 mixture ≠ sum(stems)，
    # 前端"独奏四轨 == 播放混音"这条不变量就破了。
    peak = float(np.max(np.abs(mixture)))
    if peak > 0.95:
        scale = 0.95 / peak
        stems = {k: (v * scale).astype(np.float32) for k, v in stems.items()}
        mixture = (mixture * scale).astype(np.float32)
        print(f"  峰值 {peak:.3f} → 整体缩放 ×{scale:.3f}")

    # 注：mp3 是有损编码，孤立的单样本瞬态峰值解码后会被削掉 2~3 dB（这里 0.95 → ~0.7）。
    # 属正常现象，不影响听感与 RMS，也不破坏 mixture = Σstems 这条不变量。
    residual = float(np.max(np.abs(mixture - sum(stems.values()))))
    print(f"  自检：|mixture - Σstems| 最大 = {residual:.2e}（应当接近 0）")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / "_tmp.wav"

    for name, audio in [("mixture", mixture), *stems.items()]:
        save_audio(tmp, audio, SR, subtype="PCM_16")
        mp3 = OUT_DIR / f"{name}.mp3"
        if encode_mp3(tmp, mp3):
            print(f"  ✅ {mp3}  ({mp3.stat().st_size / 1024:.0f} KB)")
        else:
            wav = OUT_DIR / f"{name}.wav"
            save_audio(wav, audio, SR, subtype="PCM_16")
            print(f"  ⚠️  ffmpeg 编码失败，改存 {wav}")
    tmp.unlink(missing_ok=True)

    analysis = OUT_DIR / "analysis.json"
    analysis.write_text(json.dumps(build_analysis(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ {analysis}（BPM/拍点/和弦/段落，全部是真值）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
