"""和弦识别：拍同步 chroma + 三和弦模板匹配。

流程：拍点 → 每拍取 chroma 中位数（拍同步）→ 与 24 个三和弦模板相关 →
中值滤波去抖 → 合并相邻同名和弦成区间。

**为什么用拍同步而不是固定窗口**：和弦变化几乎总是发生在拍点上。
按拍聚合既降噪又让边界天然落在音乐位置上，
而固定窗口会把一个和弦切成互相矛盾的碎片。

.. note::
   只识别**大三和弦与小三和弦**（24 类）。七和弦、挂留、转位一律
   归到最接近的三和弦上。这是刻意的取舍：模板法在更细的和弦集上
   会迅速退化，而 24 类已经够驱动混音台的和弦轨。
   要做真正的大词表和弦识别得上训练模型，那是另一个量级的工作。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.analysis.key import PITCH_NAMES


@dataclass
class Chord:
    start: float
    end: float
    label: str          # 如 "Am"、"F"
    confidence: float

    def as_dict(self) -> dict:
        return {"start": round(float(self.start), 3), "end": round(float(self.end), 3),
                "label": self.label, "confidence": round(float(self.confidence), 3)}


def _templates() -> tuple[np.ndarray, list[str]]:
    """24 个三和弦模板：根音+大三度+纯五度（大），根音+小三度+纯五度（小）。"""
    tpl, names = [], []
    for root in range(12):
        for third, suffix in ((4, ""), (3, "m")):
            v = np.zeros(12)
            v[root] = v[(root + third) % 12] = v[(root + 7) % 12] = 1.0
            tpl.append(v / np.linalg.norm(v))
            names.append(f"{PITCH_NAMES[root]}{suffix}")
    return np.stack(tpl), names


def _median_filter(idx: np.ndarray, k: int = 3) -> np.ndarray:
    """对和弦序列做中值滤波，压掉单拍的抖动。

    和弦不会只持续一拍就换掉再换回来 —— 那种模式几乎总是噪声。
    """
    if k < 3 or len(idx) < k:
        return idx
    out = idx.copy()
    h = k // 2
    for i in range(h, len(idx) - h):
        window = idx[i - h : i + h + 1]
        vals, counts = np.unique(window, return_counts=True)
        out[i] = vals[np.argmax(counts)]
    return out


def analyze_chords(y: np.ndarray, sr: int, beats: np.ndarray,
                   smooth: int = 3) -> list[Chord]:
    """在给定拍点上识别和弦。

    Args:
        beats: 拍点时刻（秒）。来自 :func:`src.analysis.beat.analyze_beats`
        smooth: 中值滤波窗口（拍）。0 或 1 表示不滤波
    """
    import librosa

    if len(beats) < 2:
        return []

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    frames = librosa.time_to_frames(beats, sr=sr)
    frames = np.clip(frames, 0, chroma.shape[1] - 1)
    # 拍同步：每拍内取中位数，比均值更抗瞬态尖峰
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)

    tpl, names = _templates()
    # 逐拍归一化后与模板做内积 = 余弦相似度
    norm = sync / (np.linalg.norm(sync, axis=0, keepdims=True) + 1e-12)
    sim = tpl @ norm                       # (24, n_beats)
    idx = np.argmax(sim, axis=0)
    conf = sim[idx, np.arange(sim.shape[1])]
    idx = _median_filter(idx, smooth)

    # 合并相邻的同名和弦
    out: list[Chord] = []
    n = min(len(idx), len(beats) - 1)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and idx[j + 1] == idx[i]:
            j += 1
        out.append(Chord(start=float(beats[i]), end=float(beats[j + 1]),
                         label=names[idx[i]], confidence=float(conf[i : j + 1].mean())))
        i = j + 1
    return out


def to_mir_eval_label(label: str) -> str:
    """把展示用的 "Am" / "F" 转成 mir_eval 的 "A:min" / "F:maj"。

    两套记法**刻意不统一**：界面上给人看的是乐手习惯的 "Am"，
    评测走的是 mir_eval 的标准语法。若为了省事在界面上写 "A:min"，
    就是让工具的实现细节渗进产品；反过来把 "Am" 塞给 mir_eval 会直接抛异常
    （这个异常已经在第一次跑评测时抓到过）。
    """
    if not label or label in ("N", "X"):
        return "N"
    if label.endswith("m"):
        return f"{label[:-1]}:min"
    return f"{label}:maj"


def evaluate_chords(ref: list[dict], est: list[Chord]) -> dict[str, float]:
    """和弦识别的加权准确率（按时间加权）。

    用 mir_eval 的 majmin 口径：把两边都映射到大/小三和弦再比，
    这与本模块只输出 24 类的设定一致。
    """
    import mir_eval

    if not ref or not est:
        return {"weighted_accuracy": 0.0, "overlap_seconds": 0.0}

    r_int = np.array([[c["start"], c["end"]] for c in ref])
    r_lab = [to_mir_eval_label(c["label"]) for c in ref]
    e_int = np.array([[c.start, c.end] for c in est])
    e_lab = [to_mir_eval_label(c.label) for c in est]

    # mir_eval 要求区间连续且覆盖同一段时间，先取交集再对齐
    lo = max(r_int[0, 0], e_int[0, 0])
    hi = min(r_int[-1, 1], e_int[-1, 1])
    if hi <= lo:
        return {"weighted_accuracy": 0.0, "overlap_seconds": 0.0}

    r_int, r_lab = mir_eval.util.adjust_intervals(r_int, r_lab, lo, hi, "N")
    e_int, e_lab = mir_eval.util.adjust_intervals(e_int, e_lab, lo, hi, "N")
    intervals, rl, el = mir_eval.util.merge_labeled_intervals(r_int, r_lab, e_int, e_lab)
    durations = intervals[:, 1] - intervals[:, 0]
    comparisons = mir_eval.chord.majmin(rl, el)
    return {
        "weighted_accuracy": float(mir_eval.chord.weighted_accuracy(comparisons, durations)),
        "overlap_seconds": float(hi - lo),
    }
