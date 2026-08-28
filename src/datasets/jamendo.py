"""MTG-Jamendo 数据访问（P4 标签任务）。

目录结构::

    data/jamendo/
    ├── meta/
    │   ├── autotagging_real.tsv          # 全量标注（注意不是 autotagging.tsv，见下）
    │   ├── autotagging_instrument.tsv
    │   ├── autotagging_top50tags.tsv
    │   └── splits/split-0/{subset}-{train,validation,test}.tsv
    └── audio/
        └── NN/<track_id>.mp3             # NN = 曲目 id 的末两位

.. warning::
   **``autotagging.tsv`` 是个 31 字节的指针文件**，内容只有一行真实文件名
   （``raw_30s_cleantags_50artists.tsv``），不是数据本身。
   直接当 tsv 读会得到一行垃圾且不报错。下载脚本已经把真实文件另存为
   ``autotagging_real.tsv``，这里只读后者。

.. note::
   **本地只有全量的一个子样本。** 数据集按曲目 id 末两位分成 100 块，
   我们只下了前 N 块（实测每块 556±22 首，是干净的随机样本）。
   所以所有 split 文件都要**按本地实际存在的曲目过滤**，
   否则训练集里会有大量指向不存在文件的条目。:func:`load_split` 默认就这么做。

许可：元数据 CC BY-NC-SA 4.0，**仅限非商业研究与学术使用**；音频不可再分发。
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("data/jamendo")
CATEGORIES = ("genre", "instrument", "mood/theme")


@dataclass
class Track:
    """一条标注记录。``path`` 形如 ``"14/214.mp3"``。"""

    track_id: str
    artist_id: str
    album_id: str
    path: str
    duration: float
    tags: tuple[str, ...]

    @property
    def chunk(self) -> int:
        return int(self.path.split("/")[0])

    def audio_path(self, root: Path = DEFAULT_ROOT, dtype: str = "audio") -> Path:
        # audio-low 的文件名多了 ".low"
        p = self.path.replace(".mp3", ".low.mp3") if dtype == "audio-low" else self.path
        return root / dtype / p


def _read_tsv(path: Path) -> list[Track]:
    out = []
    with open(path, encoding="utf-8") as f:
        header = next(f, "")
        if not header.startswith("TRACK_ID"):
            raise ValueError(
                f"{path} 看起来不是标注文件（首行：{header[:60]!r}）。\n"
                "如果它只有一行文件名，说明读到了 autotagging.tsv 这个**指针文件** —— "
                "改读 autotagging_real.tsv。"
            )
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 6:
                continue
            out.append(Track(p[0], p[1], p[2], p[3], float(p[4]), tuple(t for t in p[5:] if t)))
    return out


@dataclass
class TagVocab:
    """标签词表：名字 ↔ 下标，以及按类别的分组。

    ``groups`` 直接喂给 :func:`src.eval.tagging.evaluate_tagging` 的 ``tag_groups``，
    **M5 的 H2 假设检验（stem-aware 是否主要惠及乐器标签）就靠它出分类别指标。**
    """

    tags: tuple[str, ...]
    index: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        self.index = {t: i for i, t in enumerate(self.tags)}

    def __len__(self) -> int:
        return len(self.tags)

    @property
    def groups(self) -> dict[str, np.ndarray]:
        g = collections.defaultdict(list)
        for i, t in enumerate(self.tags):
            g[t.split("---")[0]].append(i)
        return {k: np.array(v) for k, v in sorted(g.items())}

    def encode(self, tracks: list[Track]) -> np.ndarray:
        """→ ``(n_tracks, n_tags)`` 的 0/1 矩阵（float32，BCE 直接能用）。"""
        y = np.zeros((len(tracks), len(self.tags)), dtype=np.float32)
        for i, t in enumerate(tracks):
            for tag in t.tags:
                j = self.index.get(tag)
                if j is not None:
                    y[i, j] = 1.0
        return y


def available_chunks(root: Path = DEFAULT_ROOT, dtype: str = "audio") -> set[int]:
    """本地实际存在哪些块。"""
    d = root / dtype
    if not d.is_dir():
        return set()
    return {int(x.name) for x in d.iterdir()
            if x.is_dir() and len(x.name) == 2 and x.name.isdigit() and any(x.iterdir())}


def load_subset(
    subset: str = "autotagging_real",
    root: Path = DEFAULT_ROOT,
    only_local: bool = True,
    dtype: str = "audio",
) -> list[Track]:
    """读一个标注文件。``only_local=True`` 时只保留本地已下载的块。"""
    tracks = _read_tsv(Path(root) / "meta" / f"{subset}.tsv")
    if not only_local:
        return tracks
    have = available_chunks(Path(root), dtype)
    if not have:
        raise FileNotFoundError(
            f"{Path(root) / dtype} 下没有任何音频块。\n"
            f"先跑：python -m scripts.download_jamendo --chunks 20 --type {dtype}"
        )
    return [t for t in tracks if t.chunk in have]


def load_split(
    subset: str = "autotagging",
    split: int = 0,
    root: Path = DEFAULT_ROOT,
    only_local: bool = True,
    dtype: str = "audio",
    min_positives: int = 0,
) -> tuple[dict[str, list[Track]], TagVocab]:
    """读官方 split-N 的 train/validation/test。

    Args:
        min_positives: 丢掉在**训练集**里正样本少于该数的标签。
            这不是可选的美化 —— 正样本个位数的标签上，
            AP 不稳定、逐标签阈值搜索纯属过拟合噪声，
            留着它们只会让 Macro-F1 变成噪声的平均。
            本地只有子样本时尤其重要（10 块下有 81/195 个标签的验证集正样本不足 10）。

    Returns:
        ``({"train": [...], "validation": [...], "test": [...]}, TagVocab)``
        —— 词表由**训练集**决定，验证/测试集里的其他标签会被忽略。
    """
    root = Path(root)
    parts = {}
    for name in ("train", "validation", "test"):
        parts[name] = _read_tsv(root / "meta" / "splits" / f"split-{split}" / f"{subset}-{name}.tsv")

    if only_local:
        have = available_chunks(root, dtype)
        if not have:
            raise FileNotFoundError(
                f"{root / dtype} 下没有任何音频块。\n"
                f"先跑：python -m scripts.download_jamendo --chunks 20 --type {dtype}"
            )
        parts = {k: [t for t in v if t.chunk in have] for k, v in parts.items()}

    counts = collections.Counter(tag for t in parts["train"] for tag in t.tags)
    tags = tuple(sorted(t for t, n in counts.items() if n >= min_positives))
    return parts, TagVocab(tags)


def tag_statistics(tracks: list[Track], vocab: TagVocab | None = None) -> dict:
    """标签频次、长尾程度、每首歌的平均标签数。画长尾分布图和写报告都用它。"""
    counts = collections.Counter(tag for t in tracks for tag in t.tags)
    if vocab is not None:
        counts = collections.Counter({k: v for k, v in counts.items() if k in vocab.index})

    n = len(tracks)
    freq = np.array(sorted(counts.values())[::-1]) if counts else np.array([0])
    per_cat = collections.defaultdict(list)
    for tag, c in counts.items():
        per_cat[tag.split("---")[0]].append(c)

    return {
        "n_tracks": n,
        "n_tags": len(counts),
        "total_hours": float(sum(t.duration for t in tracks) / 3600),
        "tags_per_track": float(np.mean([len(t.tags) for t in tracks])) if tracks else 0.0,
        "label_density": float(freq.sum() / (n * len(counts))) if counts and n else 0.0,
        "freq_max": int(freq.max()),
        "freq_median": int(np.median(freq)),
        "freq_min": int(freq.min()),
        # 头部 10% 的标签占了多少比例的正样本 —— 长尾程度的单数字刻画
        "head10pct_share": float(freq[: max(1, len(freq) // 10)].sum() / freq.sum()) if freq.sum() else 0.0,
        "n_below_50": int((freq < 50).sum()),
        "by_category": {
            c: {"n_tags": len(v), "min": int(min(v)), "median": int(np.median(v)), "max": int(max(v))}
            for c, v in sorted(per_cat.items())
        },
        "counts": dict(counts.most_common()),
    }
