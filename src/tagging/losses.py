"""多标签分类的损失函数（P4-L3 的对比项）。

标签矩阵密度只有 2.15%（实测，见 ``results/p4_tag_distribution.md``）——
**97.9% 的位置是 0**。普通 BCE 会被海量负样本主导，模型学到"全预测 0"就能拿到
很低的 loss，稀有标签直接被放弃。这一层是 L3 要解决的问题。

三个候选，各自的思路不同：

- :class:`BCELoss` —— 基线，什么都不做
- :class:`FocalLoss` —— 降低"已经学会的容易样本"的权重，逼模型看难样本
- :class:`AsymmetricLoss` —— **专为多标签设计**：正负样本用不同的调制，
  并且给负样本加一个概率下限（probability shifting），
  把"模型已经很确定是负"的样本彻底丢弃。多标签任务上通常优于 Focal。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """标准逐标签二元交叉熵。多标签的默认选择，也是本项目的基线。

    Args:
        pos_weight: 每个标签的正样本权重，形状 ``(n_tags,)``。
            传 ``n_neg/n_pos`` 是最朴素的不平衡处理 —— 但它对极稀有标签会给出
            极大的权重（几百倍），反而让训练不稳，所以本项目默认不用，
            只作为消融的一项。
    """

    def __init__(self, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight)


class FocalLoss(nn.Module):
    r"""Focal Loss（Lin et al. 2017）。

    .. math:: \mathrm{FL}(p_t) = -\alpha (1-p_t)^\gamma \log(p_t)

    ``(1-p_t)^gamma`` 这一项让**已经预测对的样本**贡献的梯度快速衰减，
    于是训练的注意力自动转向难样本。

    Args:
        gamma: 越大越激进。2.0 是论文默认。
        alpha: 正样本的额外权重；None 表示不加。
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce * (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            a_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = a_t * loss
        return loss.mean()


class AsymmetricLoss(nn.Module):
    r"""Asymmetric Loss（Ridnik et al. 2021），专为**多标签**设计。

    相对 Focal 的两个关键改动：

    1. **正负样本用不同的 gamma**（``gamma_pos`` 通常 0，``gamma_neg`` 通常 4）。
       多标签里正样本本来就稀缺，不该再降它们的权重 —— Focal 对正负一视同仁
       是它在多标签上不如 ASL 的主要原因。
    2. **概率平移（probability shifting）**：把负样本的预测概率减去 ``clip``
       再截断到 0。效果是**彻底丢弃**那些模型已经很确定是负的样本
       （``p < clip`` 的直接不产生梯度），而不只是降权。

    Args:
        gamma_neg / gamma_pos: 负/正样本的调制指数。
        clip: 负样本概率平移量，0.05 是论文默认；设 0 则退化为非对称 Focal。
    """

    def __init__(self, gamma_neg: float = 4.0, gamma_pos: float = 0.0,
                 clip: float = 0.05, eps: float = 1e-8):
        super().__init__()
        self.gamma_neg, self.gamma_pos = gamma_neg, gamma_pos
        self.clip, self.eps = clip, eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        p_pos = p
        p_neg = (1 - p + self.clip).clamp(max=1.0) if self.clip > 0 else 1 - p

        loss_pos = targets * torch.log(p_pos.clamp(min=self.eps))
        loss_neg = (1 - targets) * torch.log(p_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        # 调制项不参与反传：论文的做法，能让训练明显更稳
        with torch.no_grad():
            pt = p_pos * targets + (1 - p_neg) * (1 - targets)
            gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
            w = (1 - pt).pow(gamma)
        return -(loss * w).mean()


def build_loss(name: str, **kw) -> nn.Module:
    """``"bce"`` / ``"focal"`` / ``"asl"``。消融脚本按名字取。"""
    table = {"bce": BCELoss, "focal": FocalLoss, "asl": AsymmetricLoss}
    if name not in table:
        raise ValueError(f"未知损失 {name!r}，可选：{list(table)}")
    return table[name](**kw)
