"""标签模型与损失的单测。

模型部分只能验证**结构性质**（形状、参数量、梯度能不能回传）。
损失部分则有真正的行为断言 —— 那几条才是 L3 消融成立的前提。
"""

import numpy as np
import pytest
import torch

from src.tagging.features import MelConfig, normalize_stats
from src.tagging.losses import AsymmetricLoss, BCELoss, FocalLoss, build_loss
from src.tagging.models import (
    AttentionPool,
    AttentionPoolHead,
    LinearProbe,
    MelCNN,
    count_params,
    pool_frames,
)

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# 模型形状与梯度
# --------------------------------------------------------------------------------------

def test_melcnn_shapes_and_size():
    m = MelCNN(n_tags=50)
    x = torch.randn(3, 128, 512)
    assert m(x).shape == (3, 50)
    # L0 是下界参照，刻意保持小；大了就模糊了阶梯的含义
    assert count_params(m) < 1_200_000, f"MelCNN 有 {count_params(m):,} 参数，太大了"


def test_melcnn_backward():
    m = MelCNN(n_tags=10)
    loss = m(torch.randn(2, 128, 256)).sum()
    loss.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_linear_probe_shapes():
    m = LinearProbe(dim=768, n_tags=50)
    assert m(torch.randn(4, 120, 768)).shape == (4, 50)


def test_attention_head_shapes():
    m = AttentionPoolHead(dim=768, n_tags=50)
    assert m(torch.randn(4, 120, 768)).shape == (4, 50)


@pytest.mark.parametrize("pooling", ["mean", "max", "attention"])
def test_head_supports_all_poolings(pooling):
    m = AttentionPoolHead(dim=64, n_tags=7, pooling=pooling)
    assert m(torch.randn(2, 30, 64)).shape == (2, 7)


# --------------------------------------------------------------------------------------
# 注意力池化：L2 存在的理由
# --------------------------------------------------------------------------------------

def test_attention_pool_can_ignore_most_frames():
    """**注意力池化的核心能力**：把权重集中到少数帧上。

    构造一个 100 帧的序列，只有 3 帧带信号，其余是噪声。
    平均池化会把信号稀释到 3%；注意力池化**有能力**（不是必然，需要训练）
    把它保留下来。这里训练几十步验证这个能力确实存在 ——
    如果连这个玩具任务都学不会，说明池化模块写错了。

    这正是乐器标签的处境：一段只在副歌出现 8 秒的电吉他，
    在 3 分钟的曲子里占比不到 5%。
    """
    T, D, signal_frames = 100, 16, [10, 11, 12]
    pool = AttentionPool(D)
    proj = torch.nn.Linear(D, 1)
    opt = torch.optim.Adam(list(pool.parameters()) + list(proj.parameters()), lr=0.05)

    def batch(n=32):
        x = torch.randn(n, T, D) * 0.1
        y = torch.randint(0, 2, (n, 1)).float()
        # 正样本才在那 3 帧上放信号
        x[:, signal_frames, 0] += y * 5.0
        return x, y

    for _ in range(120):
        x, y = batch()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(proj(pool(x)), y)
        opt.zero_grad(); loss.backward(); opt.step()

    x, y = batch(256)
    with torch.no_grad():
        acc = ((proj(pool(x)) > 0).float() == y).float().mean().item()
    assert acc > 0.9, f"注意力池化在玩具任务上只有 {acc:.2f} 准确率，池化模块可能有问题"


def test_pool_frames_modes_differ():
    x = torch.randn(2, 20, 8)
    assert not torch.allclose(pool_frames(x, "mean"), pool_frames(x, "max"))


def test_pool_frames_rejects_unknown():
    with pytest.raises(ValueError):
        pool_frames(torch.randn(1, 2, 3), "median")


# --------------------------------------------------------------------------------------
# 损失：L3 消融成立的前提
# --------------------------------------------------------------------------------------

def _logits_targets(n=64, t=20, pos_rate=0.02, seed=0):
    """构造一个和真实数据密度相仿的稀疏多标签批次（实测密度 2.15%）。"""
    g = torch.Generator().manual_seed(seed)
    targets = (torch.rand(n, t, generator=g) < pos_rate).float()
    logits = torch.randn(n, t, generator=g)
    return logits, targets


def test_all_losses_are_positive_and_backward():
    logits, targets = _logits_targets()
    for name in ("bce", "focal", "asl"):
        loss = build_loss(name)(logits.requires_grad_(True), targets)
        assert loss.item() > 0
        loss.backward()


def test_focal_downweights_easy_samples():
    """Focal 的定义性质：预测已经很准时，它的 loss 相对 BCE 被压得更低。"""
    targets = torch.tensor([[1.0, 0.0]])
    easy = torch.tensor([[6.0, -6.0]])     # 预测非常准
    hard = torch.tensor([[0.1, -0.1]])     # 预测很含糊

    bce, focal = BCELoss(), FocalLoss(gamma=2.0)
    ratio_easy = focal(easy, targets) / bce(easy, targets)
    ratio_hard = focal(hard, targets) / bce(hard, targets)
    assert ratio_easy < ratio_hard, "Focal 没有相对地压低容易样本"


def test_asl_discards_confident_negatives():
    """ASL 的概率平移：模型已经很确定是负的样本应当**完全不产生梯度**。

    这是 ASL 相对 Focal 的关键差异 —— 不是降权，是彻底丢弃。
    """
    targets = torch.zeros(1, 1)
    very_negative = torch.tensor([[-8.0]], requires_grad=True)   # sigmoid ≈ 0.0003 < clip=0.05

    asl = AsymmetricLoss(clip=0.05)
    asl(very_negative, targets).backward()
    assert very_negative.grad.abs().item() < 1e-6, "确定的负样本仍在产生梯度"


def test_asl_keeps_gradient_on_uncertain_negatives():
    targets = torch.zeros(1, 1)
    uncertain = torch.tensor([[0.5]], requires_grad=True)        # sigmoid ≈ 0.62 ≫ clip
    AsymmetricLoss(clip=0.05)(uncertain, targets).backward()
    assert uncertain.grad.abs().item() > 1e-3


def test_asl_treats_positives_and_negatives_differently():
    """非对称性：把同样的"预测错得一样多"分别放在正/负样本上，loss 应当不同。"""
    asl = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0)
    on_pos = asl(torch.tensor([[-1.0]]), torch.ones(1, 1))
    on_neg = asl(torch.tensor([[1.0]]), torch.zeros(1, 1))
    assert not torch.isclose(on_pos, on_neg), "gamma_pos != gamma_neg 却给出了相同的 loss"


def test_bce_pos_weight_raises_positive_loss():
    logits, targets = _logits_targets(pos_rate=0.1)
    plain = BCELoss()(logits, targets)
    weighted = BCELoss(pos_weight=torch.full((targets.shape[1],), 5.0))(logits, targets)
    assert weighted > plain


def test_build_loss_rejects_unknown():
    with pytest.raises(ValueError, match="未知损失"):
        build_loss("hinge")


# --------------------------------------------------------------------------------------
# 特征配置
# --------------------------------------------------------------------------------------

def test_mel_config_frames_and_tag():
    cfg = MelConfig()
    assert cfg.n_frames == 1876
    assert cfg.tag() == "mel128_sr16000_hop256_30s"


def test_mel_config_tag_changes_with_params():
    """参数一变缓存目录就要变，否则会读到用旧参数算的特征 —— 而且不报错。"""
    assert MelConfig(n_mels=96).tag() != MelConfig().tag()
    assert MelConfig(sr=22050).tag() != MelConfig().tag()


def test_normalize_stats_uses_given_data_only():
    a = [np.full((4, 10), 2.0, np.float16)]
    mean, std = normalize_stats(a)
    assert mean == pytest.approx(2.0)
    assert std < 1e-3          # 常数输入 → 标准差接近 0，但被 eps 兜住不为 0
    assert std > 0
