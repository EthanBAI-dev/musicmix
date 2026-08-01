"""推理期后处理的单测。

这些手段都不重训模型，所以正确性只能靠**性质**来验证：
- 软掩码之后四轨之和必须精确等于混音（掩码逐点和为 1）
- 用**真值**当输入时，细化不该把结果搞坏（退化为 IRM oracle）
- TTA 的变换必须是自逆的，且对真实分离结果同变
"""

import numpy as np
import pytest

from src.datasets.synthetic import make_synthetic_track
from src.eval.separation import global_sdr
from src.separation.postprocess import (
    TTA_TRANSFORMS,
    apply_tta,
    ensemble,
    multichannel_wiener,
    soft_mask_refine,
    sum_error,
)

RNG = np.random.default_rng(20260801)
STEMS = ("vocals", "drums", "bass", "other")


def _degraded(refs: dict, mixture: np.ndarray, snr_db: float = 6.0) -> dict:
    """构造一组"有误差但不离谱"的估计，模拟真实模型输出。"""
    out = {}
    for s, y in refs.items():
        noise = RNG.standard_normal(y.shape).astype(np.float32)
        scale = np.sqrt(np.sum(y**2) / (10 ** (snr_db / 10)) / np.sum(noise**2))
        out[s] = (y + noise * scale).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------
# 软掩码细化
# --------------------------------------------------------------------------------------

def test_soft_mask_makes_stems_sum_to_mixture():
    """**软掩码最硬的性质**：掩码逐点求和为 1，所以四轨之和必然等于混音。

    这也是判断"细化到底有没有生效"的自检 —— 原始 Demucs 输出不满足这一条。
    """
    refs, mixture = make_synthetic_track(0, 4.0)
    est = _degraded(refs, mixture)

    before = sum_error(mixture, est)
    after = sum_error(mixture, soft_mask_refine(est, mixture))

    assert before > 0.01, "构造的估计本来就满足和为混音，这个用例没有意义"
    # iSTFT 边界效应会留一点残差，用相对量判断
    assert after < 0.02 * np.max(np.abs(mixture)), f"细化后仍有 {after:.4f} 的偏差"
    assert after < before


def test_soft_mask_with_true_magnitudes_equals_irm_oracle():
    """把**真值**喂进去，软掩码就退化成 IRM oracle。

    这条把 postprocess 和 eval 里两份独立实现对上了，任何一边写错都会被抓到。
    """
    from src.eval.separation import ideal_ratio_mask

    refs, mixture = make_synthetic_track(1, 3.0)
    a = soft_mask_refine(refs, mixture, alpha=2.0)
    b = ideal_ratio_mask(refs, mixture, power=2.0)

    for s in refs:
        n = min(a[s].shape[0], b[s].shape[0])
        assert np.allclose(a[s][:n], b[s][:n], atol=1e-5), f"{s} 两份实现结果不一致"


def test_soft_mask_improves_noisy_estimates():
    """对含噪估计做细化应当提升 SDR —— 噪声里超出混音张成空间的部分被抹掉了。"""
    refs, mixture = make_synthetic_track(2, 4.0)
    est = _degraded(refs, mixture, snr_db=4.0)
    ref_out = soft_mask_refine(est, mixture)

    before = np.mean([global_sdr(refs[s], est[s][: refs[s].shape[0]]) for s in refs])
    after = np.mean([global_sdr(refs[s], ref_out[s][: refs[s].shape[0]]) for s in refs])
    assert after > before, f"细化后 {after:.2f} 反而不如细化前 {before:.2f} dB"


def test_soft_mask_alpha_affects_result():
    refs, mixture = make_synthetic_track(3, 2.0)
    est = _degraded(refs, mixture)
    a1 = soft_mask_refine(est, mixture, alpha=1.0)
    a2 = soft_mask_refine(est, mixture, alpha=2.0)
    assert not np.allclose(a1["vocals"], a2["vocals"], atol=1e-4)


# --------------------------------------------------------------------------------------
# 多通道维纳滤波
# --------------------------------------------------------------------------------------

def test_mwf_preserves_shape_and_improves():
    refs, mixture = make_synthetic_track(4, 4.0)
    est = _degraded(refs, mixture, snr_db=4.0)
    out = multichannel_wiener(est, mixture)

    for s in refs:
        assert out[s].shape == mixture.shape

    before = np.mean([global_sdr(refs[s], est[s][: refs[s].shape[0]]) for s in refs])
    after = np.mean([global_sdr(refs[s], out[s][: refs[s].shape[0]]) for s in refs])
    assert after > before, f"MWF 后 {after:.2f} 反而不如之前 {before:.2f} dB"


def test_mwf_rejects_mono():
    refs, mixture = make_synthetic_track(5, 1.0)
    mono = mixture[:, :1]
    with pytest.raises(ValueError):
        multichannel_wiener({k: v[:, :1] for k, v in refs.items()}, mono)


# --------------------------------------------------------------------------------------
# TTA
# --------------------------------------------------------------------------------------

def test_tta_transforms_are_self_inverse():
    """每个变换做两次必须回到原样 —— :func:`apply_tta` 的逆变换直接依赖这一点。"""
    x = RNG.standard_normal((1000, 2)).astype(np.float32)
    for name, t in TTA_TRANSFORMS.items():
        assert np.allclose(t(t(x)), x), f"{name} 不是自逆变换"


def test_tta_on_equivariant_model_is_exact():
    """对一个**真正同变**的分离器，TTA 平均必须精确等于单次推理。

    这条钉住的是"逆变换有没有配对错"—— 如果逆变换写错，
    swap/flip 的结果会互相抵消，平均后能量明显偏低。
    """
    refs, mixture = make_synthetic_track(6, 2.0)

    def equivariant(mix):
        # 一个线性且同变的假分离器：按固定比例切分
        return {s: mix * w for s, w in zip(STEMS, (0.4, 0.3, 0.2, 0.1), strict=True)}

    single = equivariant(mixture)
    averaged = apply_tta(equivariant, mixture, ("identity", "swap", "flip", "swap_flip"))
    for s in STEMS:
        assert np.allclose(single[s], averaged[s], atol=1e-6), f"{s} TTA 平均后变了"


def test_tta_averaging_reduces_independent_noise():
    """分离器每次带**独立**噪声时，N 次平均应把噪声方差压到约 1/N。

    这是 TTA 能白捡增益的全部原理。用 identity 变换重复 4 次即可验证，
    与同变性无关（同变性由上一条测试负责）。
    """
    refs, mixture = make_synthetic_track(7, 3.0)
    rng = np.random.default_rng(0)

    def noisy(mix):
        return {s: (refs[s][: mix.shape[0]]
                    + rng.standard_normal((mix.shape[0], 2)).astype(np.float32) * 0.02)
                for s in STEMS}

    single = noisy(mixture)
    avg = apply_tta(noisy, mixture, ("identity",) * 4)

    e1 = np.mean([np.var(single[s] - refs[s][: single[s].shape[0]]) for s in STEMS])
    e4 = np.mean([np.var(avg[s] - refs[s][: avg[s].shape[0]]) for s in STEMS])
    assert e4 == pytest.approx(e1 / 4, rel=0.3), f"4 次平均后噪声方差 {e4:.2e}，期望约 {e1 / 4:.2e}"


def test_tta_calls_separator_once_per_variant():
    refs, mixture = make_synthetic_track(8, 1.0)
    calls = {"n": 0}

    def counting(mix):
        calls["n"] += 1
        return {s: mix * 0.25 for s in STEMS}

    apply_tta(counting, mixture, ("identity", "swap", "flip"))
    assert calls["n"] == 3


# --------------------------------------------------------------------------------------
# 集成
# --------------------------------------------------------------------------------------

def test_ensemble_equal_weights_is_mean():
    a = {s: np.full((100, 2), 1.0, np.float32) for s in STEMS}
    b = {s: np.full((100, 2), 3.0, np.float32) for s in STEMS}
    out = ensemble([a, b])
    assert np.allclose(out["vocals"], 2.0)


def test_ensemble_per_stem_weights():
    a = {s: np.full((100, 2), 0.0, np.float32) for s in STEMS}
    b = {s: np.full((100, 2), 10.0, np.float32) for s in STEMS}
    out = ensemble([a, b], weights={"vocals": [1, 0], "drums": [0, 1]})
    assert np.allclose(out["vocals"], 0.0)
    assert np.allclose(out["drums"], 10.0)
    assert np.allclose(out["bass"], 5.0)      # 未指定 → 等权


def test_ensemble_normalizes_weights():
    a = {s: np.full((10, 2), 2.0, np.float32) for s in STEMS}
    b = {s: np.full((10, 2), 4.0, np.float32) for s in STEMS}
    assert np.allclose(ensemble([a, b], [3, 1])["bass"], 2.5)


def test_ensemble_rejects_empty():
    with pytest.raises(ValueError):
        ensemble([])
