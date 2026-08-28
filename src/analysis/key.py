"""调性识别：Krumhansl-Schmuckler 相关法。

做法：把整曲的 chroma 求平均得到一个 12 维音高分布，
再与 24 个调（12 大调 + 12 小调）的**调性轮廓**做相关，取最高的。

调性轮廓用 Krumhansl-Kessler 的实验数据 —— 它来自 1982 年的
「探针音」听觉实验：给被试听一段确立调性的片段，再放一个音，
让他打分「这个音有多契合」。**这组数字是人耳的实测结果，不是谁拍脑袋定的。**

.. note::
   **相对大小调（如 C 大调与 a 小调）共用同一组音**，
   仅靠音高分布区分它们本来就很难 —— 这是方法的固有限制，不是实现问题。
   所以 :func:`analyze_key` 会返回 ``relative_alternative``，
   把「另一个同样说得通的答案」一并给出。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler 探针音实验的调性轮廓（1982）
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                     2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                     2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class KeyResult:
    key: str                     # 如 "A minor"
    tonic: int                   # 0=C … 11=B
    mode: str                    # "major" / "minor"
    correlation: float           # 与最佳轮廓的相关系数
    margin: float                # 与第二名的差距 —— 小就说明不确定
    relative_alternative: str    # 相对大/小调（同样音级集合的另一个解释）

    def as_dict(self) -> dict:
        return {"key": self.key, "correlation": round(float(self.correlation), 4),
                "margin": round(float(self.margin), 4),
                "relative_alternative": self.relative_alternative}


def chroma_mean(y: np.ndarray, sr: int) -> np.ndarray:
    """整曲平均 chroma。

    用 CQT chroma 而非 STFT chroma：音高是对数刻度的，
    常 Q 变换在低频有更好的音高分辨率，对调性这种音高任务更合适。
    """
    import librosa

    c = librosa.feature.chroma_cqt(y=y, sr=sr)
    v = c.mean(axis=1)
    return v / (v.sum() + 1e-12)


def _relative(tonic: int, mode: str) -> str:
    """相对调：大调↔小调，共用同一组音级。"""
    if mode == "major":
        return f"{PITCH_NAMES[(tonic + 9) % 12]} minor"
    return f"{PITCH_NAMES[(tonic + 3) % 12]} major"


def analyze_key(y: np.ndarray, sr: int) -> KeyResult:
    v = chroma_mean(y, sr)

    scores: list[tuple[float, int, str]] = []
    for tonic in range(12):
        for mode, profile in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            # 轮廓要按主音旋转，才是「这个调」的轮廓
            rolled = np.roll(profile, tonic)
            scores.append((float(np.corrcoef(v, rolled)[0, 1]), tonic, mode))

    scores.sort(reverse=True)
    corr, tonic, mode = scores[0]
    margin = corr - scores[1][0]
    return KeyResult(key=f"{PITCH_NAMES[tonic]} {mode}", tonic=tonic, mode=mode,
                     correlation=corr, margin=margin,
                     relative_alternative=_relative(tonic, mode))
