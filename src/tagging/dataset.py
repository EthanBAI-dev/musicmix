"""PyTorch Dataset：从缓存的 mel 特征喂给模型。

训练时随机裁剪一段帧，验证/测试时用固定的中段 —— 后者是为了**可复现**：
评测结果不该随机变化。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.jamendo import Track
from src.tagging.features import MelConfig, cache_path


class MelDataset(Dataset):
    """``(mel (n_mels, T), labels (n_tags,))``

    Args:
        crop_frames: 训练时随机裁剪的帧数。None 表示用整段。
            随机裁剪同时是**数据增强** —— 同一首歌每个 epoch 看到不同片段。
        train: True 随机裁剪，False 取固定中段（保证评测可复现）。
        mean/std: 用**训练集**统计出来的归一化参数。用全体数据算就是信息泄漏。
    """

    def __init__(
        self,
        tracks: list[Track],
        labels: np.ndarray,
        root: Path,
        cfg: MelConfig = MelConfig(),
        crop_frames: int | None = 512,
        train: bool = True,
        mean: float = 0.0,
        std: float = 1.0,
    ):
        if len(tracks) != len(labels):
            raise ValueError(f"曲目数 {len(tracks)} 与标签数 {len(labels)} 不一致")
        self.tracks = tracks
        self.labels = labels.astype(np.float32)
        self.root = Path(root)
        self.cfg = cfg
        self.crop = crop_frames
        self.train = train
        self.mean, self.std = mean, std
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.tracks)

    def __getitem__(self, i: int):
        path = cache_path(self.tracks[i].path, self.root, self.cfg)
        if not path.exists():
            raise FileNotFoundError(
                f"缺少特征缓存 {path}\n先跑：python -m scripts.extract_mel")
        mel = np.load(path).astype(np.float32)

        if self.crop is not None and mel.shape[1] > self.crop:
            if self.train:
                # 每次取不同起点 —— 既是增强，也让模型见到全曲各处
                s = int(self._rng.integers(0, mel.shape[1] - self.crop + 1))
            else:
                s = (mel.shape[1] - self.crop) // 2      # 评测固定中段，可复现
            mel = mel[:, s : s + self.crop]
        elif self.crop is not None and mel.shape[1] < self.crop:
            mel = np.pad(mel, ((0, 0), (0, self.crop - mel.shape[1])))

        mel = (mel - self.mean) / self.std
        return torch.from_numpy(mel), torch.from_numpy(self.labels[i])


class FeatureDataset(Dataset):
    """冻结基座输出的逐帧特征 ``(T, D)``，给 L1/L2 用。

    与 :class:`MelDataset` 分开是因为形状约定不同：
    mel 是 ``(freq, time)``（卷积要的），基座特征是 ``(time, dim)``（注意力要的）。
    混用会静默地把时间轴和特征轴搞反 —— 模型照跑，只是永远学不好。
    """

    def __init__(self, feature_paths: list[Path], labels: np.ndarray,
                 crop_frames: int | None = None, train: bool = True):
        if len(feature_paths) != len(labels):
            raise ValueError("特征数与标签数不一致")
        self.paths = feature_paths
        self.labels = labels.astype(np.float32)
        self.crop = crop_frames
        self.train = train
        self._rng = np.random.default_rng(0)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        x = np.load(self.paths[i]).astype(np.float32)      # (T, D)
        if self.crop is not None and x.shape[0] > self.crop:
            s = (int(self._rng.integers(0, x.shape[0] - self.crop + 1))
                 if self.train else (x.shape[0] - self.crop) // 2)
            x = x[s : s + self.crop]
        return torch.from_numpy(x), torch.from_numpy(self.labels[i])


def compute_norm_stats(tracks: list[Track], root: Path, cfg: MelConfig = MelConfig(),
                       max_tracks: int = 500) -> tuple[float, float]:
    """在**训练集**上估计 mel 的均值/标准差。

    抽样 max_tracks 首就够了 —— 全量算一遍要读几 GB，而这两个数字非常稳定。
    """
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tracks))[:max_tracks]
    vals = []
    for i in idx:
        p = cache_path(tracks[int(i)].path, Path(root), cfg)
        if p.exists():
            vals.append(np.load(p).astype(np.float32).reshape(-1))
    if not vals:
        raise FileNotFoundError("没有任何特征缓存，先跑 python -m scripts.extract_mel")
    cat = np.concatenate(vals)
    return float(cat.mean()), float(cat.std() + 1e-8)


class StemFeatureDataset(Dataset):
    """混音 + 4 个分离声部的特征，堆成 ``(S, T, D)``。给 L5 用。

    每个样本读 5 个 ``.npy``。这比单来源慢 5 倍，但特征已经是 float16 且降采样到
    15 Hz，一首才 0.7 MB —— 实测 DataLoader 不是瓶颈。

    .. important::
       **来源顺序固定为 ``(mixture, vocals, drums, bass, other)``**，
       因为 gate 融合训完要按这个顺序读出权重做可解释性分析。
       顺序一变，"模型倚重哪个 stem"的结论就全错了。
    """

    SOURCES = ("mixture", "vocals", "drums", "bass", "other")

    def __init__(self, source_paths: list[list[Path]], labels: np.ndarray):
        if len(source_paths) != len(labels):
            raise ValueError("样本数与标签数不一致")
        self.paths = source_paths
        self.labels = labels.astype(np.float32)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        xs = [np.load(p).astype(np.float32) for p in self.paths[i]]
        # 各来源帧数应当一致（同一段音频、同样的 stride），但 iSTFT 边界
        # 可能差一两帧，统一截到最短，避免 stack 报形状错
        t = min(x.shape[0] for x in xs)
        return (torch.from_numpy(np.stack([x[:t] for x in xs])),
                torch.from_numpy(self.labels[i]))
