"""学生模型的单测。

重点是**形状**与**恒等性**：U-Net 的跳连尺寸对不上会直接报错（容易发现），
但补齐/裁剪写错只会让输出**短一点点**，训练照跑、损失照降，
只是永远对不齐教师 —— 那才是难查的。
"""

from __future__ import annotations

import pytest
import torch

from src.separation.student import SOURCES, StudentUNet, distill_loss


@pytest.fixture(scope="module")
def model():
    return StudentUNet(base=8, depth=3)


@pytest.mark.parametrize("length", [8192, 44100, 44100 * 2 + 137, 16000])
def test_output_length_exactly_matches_input(model, length):
    """输出长度必须**恰好**等于输入 —— 差几个样本不会报错，只会静默错位。"""
    y = model(torch.randn(1, 2, length))
    assert y.shape == (1, len(SOURCES), 2, length)


def test_batch_dimension_is_independent(model):
    """同一段音频，单独跑和放进 batch 里跑，结果必须一致。

    不一致说明有跨样本泄漏（比如误用了 BatchNorm 的批统计）。
    """
    model.eval()
    x = torch.randn(3, 2, 8192)
    with torch.no_grad():
        each = torch.cat([model(x[i : i + 1]) for i in range(3)])
        together = model(x)
    assert torch.allclose(each, together, atol=1e-4)


def test_no_batchnorm(model):
    """小 batch 下 BatchNorm 的统计量不稳，会让验证指标忽高忽低。"""
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                   for m in model.modules())


def test_stft_istft_roundtrip_is_near_lossless(model):
    """掩码全为 1（复数 1+0i）时，输出应当约等于输入 ——
    这验证 STFT/iSTFT 的形状与相位约定没搞反。"""
    x = torch.randn(1, 2, 16384) * 0.1
    spec = model.stft(x)
    f, n = spec.shape[-2], spec.shape[-1]
    # 直接把混音谱复制 4 份当作"完美掩码"的结果
    est = spec.reshape(1, 1, 4, f, n).repeat(1, len(SOURCES), 1, 1, 1)
    back = model.istft(est, length=x.shape[-1])
    for s in range(len(SOURCES)):
        assert torch.allclose(back[0, s], x[0], atol=1e-3), f"声部 {s} 往返不一致"


def test_param_count_scales_with_base():
    """base 是控制模型大小的旋钮，必须真的起作用。"""
    small, big = StudentUNet(base=8).n_params, StudentUNet(base=16).n_params
    assert big > small * 2.5


def test_student_is_much_smaller_than_teacher():
    """蒸馏的前提：学生要显著小于 42 M 的教师，否则没有意义。"""
    assert StudentUNet(base=32).n_params < 42_000_000 / 4


def test_source_order_matches_htdemucs():
    """声部顺序必须与教师一致，否则蒸馏是在让学生学错的目标。

    htdemucs 的顺序是 (drums, bass, other, vocals) —— 不是字母序，
    也不是直觉顺序，写死之前必须对照。
    """
    assert SOURCES == ("drums", "bass", "other", "vocals")


def test_distill_loss_is_zero_for_identical_output():
    y = torch.randn(1, 4, 2, 8192)
    assert distill_loss(y, y.clone()).item() == pytest.approx(0.0, abs=1e-6)


def test_distill_loss_penalises_spectral_mismatch():
    """两个波形 L1 相同、但频谱差别很大的情况，频谱项必须能区分开。"""
    t = torch.arange(8192) / 44100
    a = torch.sin(2 * torch.pi * 440 * t).reshape(1, 1, 1, -1).repeat(1, 4, 2, 1)
    b = torch.sin(2 * torch.pi * 3000 * t).reshape(1, 1, 1, -1).repeat(1, 4, 2, 1)
    assert distill_loss(a, b, alpha=1.0) > distill_loss(a, b, alpha=0.0)


def test_gradients_flow(model):
    y = model(torch.randn(1, 2, 8192))
    distill_loss(y, torch.randn_like(y)).backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    assert n_grad > 10, "大部分参数应当拿到非零梯度"
