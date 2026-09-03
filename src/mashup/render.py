"""把 Mashup 方案渲染成音频：变速 → 变调 → 对齐小节线 → 混合。

典型用法是**取 A 的人声 + B 的伴奏** —— 这正是前面 P1 分离的用处：
没有分离就只能整首叠整首，那是噪音不是 Mashup。

.. warning::
   **变速与变调都会损伤音质。** librosa 的 ``time_stretch`` / ``pitch_shift``
   是相位声码器，拉伸大了会有金属感与瞬态涂抹；``pitch_shift`` 也不做
   共振峰保持，人声移调超过两个半音就会有"花栗鼠"感。
   :mod:`src.mashup.match` 里的上限就是为此设的。

.. note::
   本模块只对**可测量的东西**做断言：拼接后小节线的对齐误差。
   "好不好听"没有真值，不产出任何质量分。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RenderReport:
    """渲染的实际代价 —— 全是能量的量。"""

    sr: int
    duration: float
    stretch_applied: float
    semitones_applied: float
    n_bars_aligned: int
    align_error_ms: float          # 对齐后两首小节线的中位误差
    donor_gain_db: float

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in vars(self).items()}


def time_stretch(y: np.ndarray, rate: float, sr: int) -> np.ndarray:
    """按 ``rate`` 变速（>1 = 变快、变短）。不改变音高。"""
    import librosa

    if abs(rate - 1.0) < 1e-4:
        return y
    return librosa.effects.time_stretch(y=np.ascontiguousarray(y), rate=rate)


def pitch_shift(y: np.ndarray, semitones: float, sr: int) -> np.ndarray:
    """移调，不改变时长。"""
    import librosa

    if abs(semitones) < 1e-3:
        return y
    return librosa.effects.pitch_shift(y=np.ascontiguousarray(y), sr=sr,
                                       n_steps=float(semitones))


def align_to_downbeat(y: np.ndarray, sr: int, src_downbeat: float,
                      dst_downbeat: float) -> np.ndarray:
    """平移 ``y``，让它的第一个小节线落在 ``dst_downbeat``。

    差半拍整个 Mashup 就垮了，所以这一步比变速变调都关键。
    平移用补零/裁剪，不做交叉淡入 —— 淡入会把第一个小节线糊掉。
    """
    shift = int(round((dst_downbeat - src_downbeat) * sr))
    if shift > 0:
        return np.concatenate([np.zeros(shift, dtype=y.dtype), y])
    if shift < 0:
        return y[-shift:]
    return y


def piecewise_stretch(y: np.ndarray, sr: int, src_downbeats: np.ndarray,
                      dst_downbeats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """**逐小节**把 donor 拉到 base 的小节位置上。

    为什么需要它：真人演奏的速度不恒定。实测两首歌的小节间隔各自有
    ±3.6% 的起伏，总跨度比"恒定速度"假设分别差 −0.9 秒和 +1.9 秒。
    **单一全局拉伸率在原理上就对不齐这样的两首歌** —— 实测 114 个小节上
    误差线性累积到 1.8 秒（误差与小节序号相关系数 −0.958）。

    做法：按小节切开，每一小节各自拉伸到目标时长，再拼回去。
    每小节的拉伸率只在全局率附近微调（±4%），所以相位声码器的瑕疵可以忽略。

    代价：小节边界处会有极短的不连续。**不做交叉淡化** ——
    淡化会把落在小节线上的鼓点糊掉，而那正是听感上最要紧的位置。

    Returns:
        ``(拉伸后的音频, 各小节线的**实测**落点秒数)``。
        落点由拼接后的实际样本数算出，不是目标值 —— 相位声码器不保证
        输出长度恰好等于请求值，把目标值当结果报出来就测不到这个误差。
    """
    n = min(len(src_downbeats), len(dst_downbeats))
    if n < 2:
        return y, np.asarray(dst_downbeats[:1], dtype=float)

    head = y[: int(src_downbeats[0] * sr)]
    out = [head]
    landed = [len(head) / sr]
    total = len(head)
    for i in range(n - 1):
        a0, a1 = src_downbeats[i], src_downbeats[i + 1]
        b0, b1 = dst_downbeats[i], dst_downbeats[i + 1]
        seg = y[int(a0 * sr): int(a1 * sr)]
        if len(seg) < 2 or b1 <= b0:
            continue
        # rate = 源时长 / 目标时长（>1 = 需要压缩）
        stretched = time_stretch(seg, rate=(a1 - a0) / (b1 - b0), sr=sr)
        out.append(stretched)
        total += len(stretched)
        landed.append(total / sr)
    if len(out) == 1:
        return y, np.asarray(dst_downbeats[:1], dtype=float)
    return np.concatenate(out), np.asarray(landed, dtype=float)


def render_mashup(base_y: np.ndarray, base_analysis: dict,
                  donor_y: np.ndarray, donor_analysis: dict,
                  plan, sr: int, donor_gain_db: float = -3.0,
                  piecewise: bool = True) -> tuple[np.ndarray, RenderReport]:
    """按 ``plan`` 把 donor 拼到 base 上。

    Args:
        base_y: 基底音频（通常是某首歌的伴奏）
        donor_y: 要叠上去的音频（通常是另一首歌的人声）
        plan: :func:`src.mashup.match.plan_mashup` 的输出
        donor_gain_db: donor 的增益。默认 −3 dB —— 人声叠在伴奏上时
            等增益会盖住伴奏
        piecewise: 逐小节对齐（见 :func:`piecewise_stretch`）。
            关掉就退回单一全局拉伸率，长曲子上会线性漂移
    """
    b_down = np.asarray(base_analysis.get("downbeats") or [0.0], dtype=float)
    d_raw = np.asarray(donor_analysis.get("downbeats") or [0.0], dtype=float)

    if piecewise and len(b_down) >= 2 and len(d_raw) >= 2:
        # 逐小节拉到 base 的小节位置上。
        # d_down 是**实测落点**，不是"应该落在哪" —— 后者恒等于 b_down，
        # 报出来的误差永远是 0，那是自己骗自己。
        y, d_down = piecewise_stretch(donor_y, sr, d_raw, b_down)
        y = align_to_downbeat(y, sr, float(d_raw[0]), float(b_down[0]))
        d_down = d_down + (float(b_down[0]) - float(d_raw[0]))
    else:
        # rate=r 之后事件时刻变成 t/r（放慢则事件推后）
        y = time_stretch(donor_y, rate=plan.stretch, sr=sr)
        d_down = d_raw / plan.stretch
        y = align_to_downbeat(y, sr, float(d_down[0]), float(b_down[0]))
        d_down = d_down - d_down[0] + b_down[0]

    y = pitch_shift(y, plan.semitones, sr)

    # 4) 长度对齐后混合
    n = len(base_y)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    mixed = base_y + y * (10 ** (donor_gain_db / 20))
    peak = np.max(np.abs(mixed))
    if peak > 1.0:
        mixed = mixed / peak          # 只在削顶时归一，避免无谓地改变响度

    # 5) 报告对齐误差 —— 这是本模块唯一能客观量化的质量指标
    m = min(len(b_down), len(d_down))
    err = float(np.median(np.abs(b_down[:m] - d_down[:m])) * 1000) if m else 0.0
    return mixed, RenderReport(
        sr=sr, duration=len(mixed) / sr,
        stretch_applied=float(plan.stretch), semitones_applied=float(plan.semitones),
        n_bars_aligned=int(m), align_error_ms=err, donor_gain_db=donor_gain_db)
