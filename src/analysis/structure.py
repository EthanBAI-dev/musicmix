"""曲式分段：用自相似矩阵 + 谱聚类找出 intro / verse / chorus 这类段落。

做法（Foote 1999 的新颖度曲线 + McFee & Ellis 的谱聚类，librosa 都有现成实现）：

1. 拍同步的 CQT + chroma 特征
2. 构造递归矩阵（recurrence matrix）—— 「第 i 拍和第 j 拍像不像」
3. 谱聚类切成 k 段，相似的段落分到同一类

**段落只有编号，没有名字。** 算法能告诉你「第 1 段和第 3 段是同一种东西」，
但没法知道那种东西叫「副歌」—— 那需要标注数据训练。
所以这里输出 ``A/B/C`` 而不是 ``verse/chorus``：

    把 A 段叫成 "chorus" 会让界面看起来更专业，
    但那是**编造的信息**。宁可显示 A/B/C 让人自己听。

唯一的例外是首尾：第一段标 ``intro``、最后一段标 ``outro``，
这两个是**位置**决定的，不是内容推断，所以可以说。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Section:
    start: float
    end: float
    label: str            # "intro" / "A" / "B" / … / "outro"
    cluster: int          # 同一 cluster = 算法认为是同一种段落

    def as_dict(self) -> dict:
        return {"start": round(float(self.start), 3), "end": round(float(self.end), 3),
                "label": self.label, "cluster": int(self.cluster)}


def analyze_structure(y: np.ndarray, sr: int, beats: np.ndarray,
                      n_sections: int = 4, min_seconds: float = 4.0) -> list[Section]:
    """把曲子切成 ``n_sections`` 类段落。

    Args:
        beats: 拍点，用于拍同步（段落边界几乎总在拍上）
        n_sections: 聚类数。**不是**段落数 —— 同一类可以出现多次
            （副歌重复正是我们想抓的）
        min_seconds: 短于此的段落并入前一段，避免碎片
    """
    import librosa
    from sklearn.cluster import SpectralClustering

    duration = len(y) / sr
    if len(beats) < n_sections * 2 or duration < min_seconds * 2:
        # 太短就别硬切 —— 返回整曲一段，比切出一堆没意义的碎片诚实
        return [Section(0.0, duration, "A", 0)]

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    frames = np.clip(librosa.time_to_frames(beats, sr=sr), 0, chroma.shape[1] - 1)
    sync = librosa.util.sync(chroma, frames, aggregate=np.median)
    if sync.shape[1] < n_sections * 2:
        return [Section(0.0, duration, "A", 0)]

    # 递归矩阵：affinity 模式给出 0~1 的相似度，正好当谱聚类的邻接矩阵
    rec = librosa.segment.recurrence_matrix(sync, mode="affinity", sym=True)
    # 加一条对角带：相邻的拍本来就该相似，否则聚类会把时间上连续的段落打散
    n = rec.shape[0]
    band = np.eye(n, k=1) + np.eye(n, k=-1)
    aff = np.maximum(rec, band * rec.max())

    k = min(n_sections, n - 1)
    labels = SpectralClustering(n_clusters=k, affinity="precomputed",
                                random_state=0, assign_labels="kmeans").fit_predict(aff)

    # 把逐拍标签合并成区间
    bounds = [0] + [i for i in range(1, len(labels)) if labels[i] != labels[i - 1]] + [len(labels)]
    raw: list[tuple[float, float, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        t0 = float(beats[a])
        t1 = float(beats[b]) if b < len(beats) else duration
        raw.append((t0, t1, int(labels[a])))

    # 合并过短的段
    merged: list[list] = []
    for t0, t1, c in raw:
        if merged and t1 - t0 < min_seconds:
            merged[-1][1] = t1
        else:
            merged.append([t0, t1, c])

    names = "ABCDEFGH"
    out = []
    # intro/outro 只在段落数 >= 3 时才用。
    # 只切出一两段时，把整首（或一半）叫「前奏」是错的 ——
    # 那不是分段结果，那是**没分出结构**，应当如实显示为 A/B。
    use_edges = len(merged) >= 3
    for i, (t0, t1, c) in enumerate(merged):
        if use_edges and i == 0:
            label = "intro"
        elif use_edges and i == len(merged) - 1:
            label = "outro"
        else:
            label = names[c % len(names)]
        out.append(Section(t0, t1, label, c))
    return out
