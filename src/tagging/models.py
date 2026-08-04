"""标签模型。对应 [00-总体大纲] 里 L0→L2 的阶梯。

- :class:`MelCNN` —— **L0**：mel + 小 CNN 从头训。最弱的一档，作用是证明流水线通了
- :class:`LinearProbe` —— **L1**：冻结基座特征 + 一个线性层。
  衡量"表征本身有多好"的标准做法
- :class:`AttentionPoolHead` —— **L2**：自己设计的注意力池化 + MLP 头

**为什么池化方式是这一层的关键。** 基座模型输出的是**逐帧**特征（3 分钟的歌有上千帧），
而标签是整曲级的。简单平均会把"只在副歌出现 8 秒的电吉他"稀释到几乎消失 ——
而这正是乐器标签最常见的情形。注意力池化让模型自己学"该看哪些帧"。
:class:`AttentionPoolHead` 支持 ``mean`` / ``max`` / ``attention`` 三种，
方便在 L2 做一次干净的对比消融。
"""

from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# L0：mel + 小 CNN
# --------------------------------------------------------------------------------------

class MelCNN(nn.Module):
    """从头训练的小型卷积网络，输入是对数 mel。

    结构是音乐标签任务的经典配置（类似 Choi et al. 的 FCN）：
    5 个 ``Conv-BN-ReLU-MaxPool`` 块，逐步把 ``(128 mel, 1876 帧)`` 压到
    ``(4, 58)`` 左右，再全局池化接分类头。

    刻意保持小（约 0.5 M 参数）：L0 的作用是**下界参照**，
    不是去和冻结基座 + 头部（L1/L2）拼效果。参数堆多了反而模糊了阶梯的含义。
    """

    def __init__(self, n_tags: int, n_mels: int = 128, channels: tuple[int, ...] = (32, 64, 96, 128, 128),
                 dropout: float = 0.25):
        super().__init__()
        blocks, c_in = [], 1
        for c in channels:
            blocks += [
                nn.Conv2d(c_in, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c_in, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, n_mels, T)`` → logits ``(B, n_tags)``。"""
        h = self.features(x.unsqueeze(1))            # (B, C, mel', T')
        # 频率轴和时间轴一起全局平均：整曲级标签不需要保留位置信息
        h = h.mean(dim=(2, 3))
        return self.head(h)


# --------------------------------------------------------------------------------------
# 池化
# --------------------------------------------------------------------------------------

class AttentionPool(nn.Module):
    """带门控的注意力池化：``(B, T, D) → (B, D)``。

    每帧算一个标量注意力分数，softmax 后加权求和。
    和简单平均的区别在于**模型可以把权重集中在少数几帧上** ——
    对"只在局部出现的乐器"至关重要。
    """

    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(64, dim // 4)
        self.score = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        w = self.score(x)                             # (B, T, 1)
        if mask is not None:
            w = w.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        w = torch.softmax(w, dim=1)
        return (x * w).sum(dim=1)


def pool_frames(x: torch.Tensor, mode: str, module: nn.Module | None = None) -> torch.Tensor:
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "max":
        return x.max(dim=1).values
    if mode == "attention":
        assert module is not None
        return module(x)
    raise ValueError(f"未知池化方式 {mode!r}")


# --------------------------------------------------------------------------------------
# L1 / L2：冻结基座之上的头部
# --------------------------------------------------------------------------------------

class LinearProbe(nn.Module):
    """**L1**：冻结基座特征做平均池化，接一个线性层。

    这是衡量表征质量的标准做法。它没有任何自研成分 —— 正因如此，
    L2 相对它的提升才能干净地归因到"池化 + 头部设计"上。
    """

    def __init__(self, dim: int, n_tags: int, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(dim, n_tags)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``(B, T, D)`` 逐帧特征。"""
        return self.fc(pool_frames(x, self.pooling))


class AttentionPoolHead(nn.Module):
    """**L2**：注意力池化 + 两层 MLP。本项目自己设计的部分。

    Args:
        pooling: ``"attention"`` / ``"mean"`` / ``"max"``。
            三种共用同一套 MLP 头，这样 L2 的消融**只变池化这一个因素**，
            归因才干净。
    """

    def __init__(self, dim: int, n_tags: int, hidden: int = 512,
                 pooling: str = "attention", dropout: float = 0.3):
        super().__init__()
        self.pooling = pooling
        self.attn = AttentionPool(dim) if pooling == "attention" else None
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_tags),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(pool_frames(x, self.pooling, self.attn))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
