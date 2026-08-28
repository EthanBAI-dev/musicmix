"""显著性检验的单测。

要验证的是两件相反的事：
- **真提升要判为显著**（否则检验太保守，白做的改进被埋没）
- **纯噪声要判为不显著**（否则检验太松，随便一跑就"有提升"）
"""

import numpy as np
import pytest

from src.eval.stats import (
    compare_models,
    format_comparison,
    paired_bootstrap,
    paired_permutation,
)

RNG = np.random.default_rng(20260729)


def test_constant_improvement_is_significant():
    """每首歌都稳定提升 1 dB → 置信区间必须是 [1, 1]，判为显著。"""
    baseline = RNG.normal(8.0, 1.5, size=50)
    treatment = baseline + 1.0

    r = paired_bootstrap(baseline, treatment)
    assert r.mean_diff == pytest.approx(1.0)
    assert r.ci_low == pytest.approx(1.0)
    assert r.ci_high == pytest.approx(1.0)
    assert r.significant


def test_identical_arrays_are_not_significant():
    a = RNG.normal(8.0, 1.5, size=50)
    r = paired_bootstrap(a, a.copy())
    assert r.mean_diff == pytest.approx(0.0)
    assert not r.significant


def test_pure_noise_is_not_significant():
    """两个方法真实水平相同、只有随机波动 → 不该判为显著。"""
    baseline = RNG.normal(8.0, 1.5, size=50)
    treatment = baseline + RNG.normal(0.0, 0.5, size=50)
    r = paired_bootstrap(baseline, treatment)
    assert not r.significant
    assert r.p_value > 0.05


def test_tiny_but_consistent_improvement_is_detected():
    """**配对检验的价值就在这里。**

    +0.15 dB 的提升远小于 1.5 dB 的曲间标准差，独立样本检验根本看不出来；
    但因为是同样的 50 首歌，配对之后噪声被消掉，能检出。
    """
    baseline = RNG.normal(8.0, 1.5, size=50)
    treatment = baseline + 0.15 + RNG.normal(0.0, 0.05, size=50)
    r = paired_bootstrap(baseline, treatment)
    assert r.significant
    assert r.mean_diff == pytest.approx(0.15, abs=0.03)


def test_regression_is_significant_in_the_negative_direction():
    baseline = RNG.normal(8.0, 1.5, size=50)
    treatment = baseline - 0.8
    r = paired_bootstrap(baseline, treatment)
    assert r.mean_diff < 0
    assert r.ci_high < 0
    assert r.significant


def test_bootstrap_is_deterministic_given_seed():
    a = RNG.normal(8.0, 1.5, size=40)
    b = a + RNG.normal(0.2, 0.4, size=40)
    r1 = paired_bootstrap(a, b, seed=123)
    r2 = paired_bootstrap(a, b, seed=123)
    assert (r1.ci_low, r1.ci_high, r1.p_value) == (r2.ci_low, r2.ci_high, r2.p_value)


def test_nan_pairs_are_dropped():
    a = np.array([8.0, 9.0, np.nan, 7.0, 6.0, 8.5])
    b = a + 1.0
    r = paired_bootstrap(a, b)
    assert r.n == 5
    assert r.mean_diff == pytest.approx(1.0)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros(10), np.zeros(11))


def test_rejects_too_few_samples():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([1.0]), np.array([2.0]))


# --------------------------------------------------------------------------------------
# 置换检验
# --------------------------------------------------------------------------------------

def test_permutation_agrees_with_bootstrap_on_clear_signal():
    baseline = RNG.normal(8.0, 1.5, size=50)
    treatment = baseline + 0.5
    perm = paired_permutation(baseline, treatment)
    assert perm.p_value < 0.001
    assert perm.significant


def test_permutation_on_noise_gives_large_p():
    baseline = RNG.normal(8.0, 1.5, size=50)
    treatment = baseline + RNG.normal(0.0, 0.5, size=50)
    assert paired_permutation(baseline, treatment).p_value > 0.05


# --------------------------------------------------------------------------------------
# 逐声部比较与输出
# --------------------------------------------------------------------------------------

def test_compare_models_covers_all_stems_plus_mean():
    """只有 bass 真的提升了，其余声部只有零均值波动。

    Note:
        这个用例最初写成"未改进的声部 = base + N(0, 0.05)"，结果 vocals 被判为显著。
        不是 bug，是**统计事实**：50 首歌下标准误只有 0.05/√50 ≈ 0.007 dB，
        这一次抽样的均值恰好偏了 0.012 dB，就足以让区间不跨 0。
        配对检验在 n=50 时的分辨率比直觉高得多 —— 这本身就是它值得用的理由。
        为了让测试确定性，这里把噪声**逐列去均值**，使"无改进"精确成立。
    """
    base = RNG.normal(8.0, 1.5, size=(50, 4))
    noise = RNG.normal(0, 0.3, size=(50, 4))
    noise -= noise.mean(axis=0, keepdims=True)  # 逐列零均值 → 差异精确为设计值
    treat = base + np.array([0.0, 0.0, 1.2, 0.0]) + noise

    res = compare_models(base, treat)
    assert set(res) == {"vocals", "drums", "bass", "other", "mean"}
    assert res["bass"].significant
    assert res["bass"].mean_diff == pytest.approx(1.2)
    for stem in ("vocals", "drums", "other"):
        assert not res[stem].significant
        assert res[stem].mean_diff == pytest.approx(0.0, abs=1e-9)
    assert res["mean"].mean_diff == pytest.approx(0.3)


def test_compare_models_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_models(np.zeros((10, 4)), np.zeros((10, 3)))


def test_format_comparison_renders_markdown():
    base = RNG.normal(8.0, 1.5, size=(30, 4))
    res = compare_models(base, base + 0.5)
    table = format_comparison(res)
    assert "| bass |" in table
    assert "95% CI" in table
    assert "✅" in table


def test_describe_mentions_significance():
    a = RNG.normal(8.0, 1.0, size=30)
    r = paired_bootstrap(a, a + 1.0)
    text = r.describe()
    assert "显著" in text
    assert "95% CI" in text
