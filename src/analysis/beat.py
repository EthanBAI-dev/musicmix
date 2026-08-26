"""节拍分析：速度（BPM）、拍点、小节线（downbeat）。

用 librosa 的动态规划节拍跟踪，不自己实现 —— 它是这一类方法的标准实现，
自己写一遍只会引入难查的偏差，而 P0 的教训是**能用标准实现就别自己写**。

小节线是自己做的：librosa 只给拍点，不给「哪一拍是第一拍」。
做法见 :func:`estimate_downbeats`。

.. note::
   **速度倍频错误（octave error）是这类算法的头号失败模式** ——
   把 96 BPM 判成 192 或 48。这不是 bug，是节拍这个量本身有歧义：
   同一段音乐说它是 96 还是 192，取决于你认为哪一层是「拍」。
   所以 :func:`analyze_beats` 会一并返回 ``tempo_candidates``，
   评测时也用 mir_eval 的 F-measure（它对倍频错误敏感，不会掩盖问题）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_SR = 22050          # 节拍分析不需要 44.1k，降采样能快一倍且不影响结果
HOP = 512


@dataclass
class BeatResult:
    """一次节拍分析的全部输出。"""

    tempo: float                                  # BPM
    beats: np.ndarray                             # 拍点时刻（秒）
    downbeats: np.ndarray                         # 小节线时刻（秒）
    beats_per_bar: int
    tempo_candidates: list[float] = field(default_factory=list)
    downbeat_confidence: float = 0.0              # 0~1，见 estimate_downbeats

    @property
    def n_bars(self) -> int:
        return len(self.downbeats)

    def as_dict(self) -> dict:
        return {
            "bpm": round(float(self.tempo), 2),
            "time_signature": f"{self.beats_per_bar}/4",
            "beats": [round(float(t), 4) for t in self.beats],
            "downbeats": [round(float(t), 4) for t in self.downbeats],
            "tempo_candidates": [round(float(t), 2) for t in self.tempo_candidates],
            "downbeat_confidence": round(float(self.downbeat_confidence), 3),
        }


def onset_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    return librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP, aggregate=np.median)


def estimate_downbeats(
    onset_env: np.ndarray,
    beats: np.ndarray,
    sr: int,
    beats_per_bar: int = 4,
) -> tuple[np.ndarray, float]:
    """在已知拍点上找出「哪一拍是小节的第一拍」。

    思路：小节线上的音符通常更重（鼓组的 kick、和弦切换都倾向落在正拍）。
    所以对 ``beats_per_bar`` 个可能的相位各算一次「该相位上所有拍的起始强度之和」，
    取最大的那个相位。

    这是个**朴素但可解释**的启发式，明确它的适用范围：

    - 前提是重音落在正拍。**切分感强的曲风（放克、雷鬼）会判错**，
      因为它们的重音故意落在反拍
    - 它只决定**相位**，不决定拍号；``beats_per_bar`` 是外部给的
    - 返回的置信度 = 最优相位得分 / 所有相位得分之和 × 相位数。
      1.0 表示各相位无差别（**完全没有把握**），越大表示越确信。
      置信度接近 1 时应当在界面上标注「不确定」，而不是假装有结论

    Returns:
        (小节线时刻, 置信度)
    """
    if len(beats) == 0:
        return np.array([]), 0.0

    frames = np.round(beats * sr / HOP).astype(int)
    frames = np.clip(frames, 0, len(onset_env) - 1)
    strength = onset_env[frames]

    scores = np.array([strength[p::beats_per_bar].sum() for p in range(beats_per_bar)])
    if scores.sum() <= 0:
        return beats[::beats_per_bar], 1.0

    phase = int(np.argmax(scores))
    # 归一化成「最优相位比平均相位强多少倍」；1.0 = 毫无区分度
    confidence = float(scores[phase] / scores.mean())
    return beats[phase::beats_per_bar], confidence


def extend_beats(beats: np.ndarray, duration: float) -> np.ndarray:
    """把拍点按中位拍长外推，补齐开头与结尾漏掉的拍。

    为什么需要：动态规划节拍跟踪在**音频最开始**往往不给拍点 ——
    它需要一段上下文才能确立周期。对 20 秒的曲子，漏掉第一拍
    意味着后续所有和弦区间整体偏移一拍（实测把和弦加权准确率从
    0.94 拉到 0.69）。

    只按**中位拍长**线性外推，不做任何花哨的事：
    外推是有风险的（前奏若是散板，补出来的拍是假的），
    所以只补到边界为止，且不改动已检测到的拍。
    """
    if len(beats) < 2:
        return beats
    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return beats

    head = []
    t = beats[0] - period
    # 补到 t >= 0 为止。**不要**用 `t > period*0.25` 这类余量守卫 ——
    # 歌曲从第一拍开始是常态，真值的首拍常常就在 0.000s，
    # 而检测出的首拍会在一拍之后，倒推回去恰好落在 0 附近，
    # 正好被余量挡掉。实测这个守卫让和弦加权准确率停在 0.688。
    while t >= 0:
        head.append(t)
        t -= period
    tail = []
    t = beats[-1] + period
    while t < duration:
        tail.append(t)
        t += period
    return np.concatenate([np.array(head[::-1]), beats, np.array(tail)])


def analyze_beats(
    y: np.ndarray,
    sr: int,
    beats_per_bar: int = 4,
    tempo_prior: float | None = None,
    extend: bool = True,
) -> BeatResult:
    """完整的节拍分析。

    Args:
        y: 单声道波形
        sr: 采样率
        beats_per_bar: 每小节几拍。默认 4/4 —— 流行音乐的绝大多数。
            **本函数不估计拍号**，见模块 docstring
        tempo_prior: 给定先验 BPM 时传入，能显著减少倍频错误
        extend: 是否把拍点外推到曲子首尾（见 :func:`extend_beats`）
    """
    import librosa

    env = onset_envelope(y, sr)
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=env, sr=sr, hop_length=HOP,
        start_bpm=tempo_prior if tempo_prior else 120.0,
        units="frames",
    )
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    if extend:
        beats = extend_beats(beats, duration=len(y) / sr)

    # 倍频候选：把这些一并报出来，让下游知道歧义在哪，而不是假装只有一个答案
    candidates = sorted({round(tempo / 2, 2), round(tempo, 2), round(tempo * 2, 2)})

    downbeats, conf = estimate_downbeats(env, beats, sr, beats_per_bar)
    return BeatResult(tempo=tempo, beats=beats, downbeats=downbeats,
                      beats_per_bar=beats_per_bar,
                      tempo_candidates=candidates, downbeat_confidence=conf)


def evaluate_beats(reference: np.ndarray, estimated: np.ndarray) -> dict[str, float]:
    """用 mir_eval 的标准口径评测拍点。

    不自己实现 F-measure：mir_eval 是 MIREX 的参考实现，
    其中 ±70 ms 容差、最小间隔等细节都有讲究，自己写很容易得出偏乐观的数字。
    """
    import mir_eval

    ref = mir_eval.beat.trim_beats(np.asarray(reference, dtype=float))
    est = mir_eval.beat.trim_beats(np.asarray(estimated, dtype=float))
    if len(ref) == 0 or len(est) == 0:
        return {"f_measure": 0.0, "cemgil": 0.0, "cmlt": 0.0, "amlt": 0.0}

    scores = mir_eval.beat.evaluate(ref, est)
    return {
        "f_measure": float(scores["F-measure"]),
        "cemgil": float(scores["Cemgil"]),
        # CMLt 要求速度与相位都对；AMLt 允许倍频/反相 ——
        # 两者的差距**就是倍频错误的量**，所以必须一起报
        "cmlt": float(scores["Correct Metric Level Total"]),
        "amlt": float(scores["Any Metric Level Total"]),
    }
