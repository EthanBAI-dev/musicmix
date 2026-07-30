"""分离评测的单测。

这些测试存在的唯一目的：**证明 SDR 没算错。**
所以每个断言的期望值都是解析上已知的，不是"跑一遍看看差不多"。
"""

import numpy as np
import pytest

from src.audio.io import match_length
from src.datasets.synthetic import make_synthetic_track
from src.eval.separation import (
    STEMS,
    TrackScores,
    aggregate,
    evaluate_track,
    format_table,
    global_sdr,
    ideal_binary_mask,
    ideal_ratio_mask,
    per_stem_matrix,
    si_sdr,
    silence_baseline,
    trivial_baseline,
)

SR = 44100
RNG = np.random.default_rng(20260729)


def _tone(freq: float, seconds: float = 4.0, amp: float = 0.3, sr: int = SR) -> np.ndarray:
    """一个立体声正弦音，左右声道略有相位差（模拟真实立体声形象）。"""
    t = np.arange(int(seconds * sr)) / sr
    left = amp * np.sin(2 * np.pi * freq * t)
    right = amp * np.sin(2 * np.pi * freq * t + 0.3)
    return np.stack([left, right], axis=1).astype(np.float32)


def _toy_song(seconds: float = 4.0, seed: int = 0):
    """一首"玩具歌"。用 :mod:`src.datasets.synthetic` 的共享实现，
    确保测试和 ``scripts.run_separation_eval --synthetic`` 跑在同一份数据上。
    """
    return make_synthetic_track(seed, seconds)


# --------------------------------------------------------------------------------------
# global SDR (uSDR)：这几条是最硬的自检
# --------------------------------------------------------------------------------------

def test_silence_estimate_is_exactly_zero_db():
    """**最重要的一条断言。**

    估计全零时 SDR 分子分母都等于 ``sum(ref^2)``，所以必然恰好 0.00 dB。
    这一条不过，:func:`global_sdr` 一定写错了，后面所有数字都不用看。
    """
    ref = _tone(440.0)
    assert global_sdr(ref, np.zeros_like(ref)) == pytest.approx(0.0, abs=1e-6)


def test_global_sdr_equals_designed_snr():
    """按设计好的信噪比构造误差，SDR 必须精确等于那个 SNR。"""
    ref = _tone(440.0)
    noise = RNG.standard_normal(ref.shape).astype(np.float32)

    for target_db in (3.0, 7.0, 15.0):
        # 把噪声能量精确缩放到 sum(ref^2) / 10^(target/10)
        scale = np.sqrt(np.sum(ref**2) / (10 ** (target_db / 10)) / np.sum(noise**2))
        est = ref + noise * scale
        assert global_sdr(ref, est) == pytest.approx(target_db, abs=1e-4)


def test_global_sdr_perfect_is_huge():
    ref = _tone(440.0)
    assert global_sdr(ref, ref.copy()) > 100.0


def test_global_sdr_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        global_sdr(_tone(440.0, 1.0), _tone(440.0, 2.0))


# --------------------------------------------------------------------------------------
# SI-SDR
# --------------------------------------------------------------------------------------

def test_si_sdr_is_scale_invariant():
    """SI-SDR 的定义性质：对整体增益免疫。

    用"有误差的估计"来验证才有意义 —— 缩放它不应该改变 SI-SDR 的值。
    """
    ref = _tone(440.0)
    est = ref + (RNG.standard_normal(ref.shape) * 0.05).astype(np.float32)

    base = si_sdr(ref, est)
    for gain in (0.1, 0.5, 2.0, 7.3):
        assert si_sdr(ref, est * gain) == pytest.approx(base, abs=1e-3)

    # 完美到只差一个增益 → 残差只剩 float32 舍入噪声，SI-SDR 应当极高
    assert si_sdr(ref, ref * 0.1) > 100.0
    assert si_sdr(ref, ref * 2.0) > 100.0  # 2 的幂次缩放无舍入误差，实际是 inf


def test_si_sdr_with_orthogonal_noise():
    """噪声与参考正交时，SI-SDR 退化为普通的能量比。"""
    ref = _tone(440.0)
    raw = RNG.standard_normal(ref.shape)
    flat_ref = ref.reshape(-1) - ref.reshape(-1).mean()
    flat_raw = raw.reshape(-1) - raw.reshape(-1).mean()
    # 把 raw 中与 ref 共线的分量投影掉
    ortho = flat_raw - (flat_raw @ flat_ref) / (flat_ref @ flat_ref) * flat_ref

    target_db = 9.0
    scale = np.sqrt((flat_ref @ flat_ref) / (10 ** (target_db / 10)) / (ortho @ ortho))
    est = (flat_ref + ortho * scale).reshape(ref.shape)
    assert si_sdr(ref, est) == pytest.approx(target_db, abs=1e-3)


def test_si_sdr_of_silent_estimate_is_negative_infinity():
    """**回归测试。** 估计全零 → SI-SDR 必须是 -inf（最差），不能是 +inf。

    早期实现里 ``noise_energy == 0`` 直接返回 +inf，没区分"完美吻合"和"什么都没输出"
    这两种同样让残差为零的情况，结果 silence 基线拿到了 +∞ 的满分。
    """
    ref = _tone(440.0)
    assert si_sdr(ref, np.zeros_like(ref)) == float("-inf")


def test_si_sdr_differs_from_global_sdr_on_gain_error():
    """一个"音量错了但内容对了"的估计：SI-SDR 应该很高，global SDR 应该很低。

    这正是保留 SI-SDR 作辅助指标的理由 —— 它能把"增益偏差"和"内容错误"分开。
    """
    ref = _tone(440.0)
    est = ref * 0.5
    assert si_sdr(ref, est) > 100.0
    assert global_sdr(ref, est) < 10.0


# --------------------------------------------------------------------------------------
# museval cSDR：验证调用与聚合没写错
# --------------------------------------------------------------------------------------

def test_museval_perfect_separation_is_high():
    refs, _ = _toy_song(seconds=3.0)
    scores = evaluate_track(refs, {k: v.copy() for k, v in refs.items()}, "perfect")
    for stem in refs:
        assert scores.csdr[stem] > 50.0, f"{stem} 完美分离却只有 {scores.csdr[stem]:.1f} dB"


def test_museval_trivial_baseline_is_low():
    """平凡基线（混音当每一轨）的 cSDR 必须很低。

    如果这里算出一个漂亮的数字，说明评测代码有问题 —— 这是 M0 的核心验收项。
    """
    refs, mixture = _toy_song(seconds=3.0)
    scores = evaluate_track(refs, trivial_baseline(mixture), "trivial")
    for stem in refs:
        assert scores.csdr[stem] < 15.0, f"{stem} 平凡基线竟有 {scores.csdr[stem]:.1f} dB"


def test_silence_baseline_usdr_is_zero_for_every_stem():
    refs, mixture = _toy_song(seconds=1.0)
    scores = evaluate_track(refs, silence_baseline(mixture), "silence", compute_csdr=False)
    for stem in refs:
        assert scores.usdr[stem] == pytest.approx(0.0, abs=1e-6)


def test_museval_failure_yields_nan_with_warning():
    """BSS Eval v4 拒绝全零估计（投影方程组欠定）。

    这不是缺陷而是它的固有约束，真实场景也会碰到：有些歌本来就没有人声轨，
    或者模型对某一轨输出了静音。要求：**记 NaN 并警告，绝不静默吞掉**，
    同时 uSDR / SI-SDR 不受影响。
    """
    refs, mixture = _toy_song(seconds=1.0)
    with pytest.warns(RuntimeWarning, match="museval"):
        scores = evaluate_track(refs, silence_baseline(mixture), "all-silent")

    assert all(np.isnan(v) for v in scores.csdr.values())
    # uSDR 这条路不经过 museval，仍然给出确定的 0.00 dB
    assert scores.usdr["vocals"] == pytest.approx(0.0, abs=1e-6)


def test_aggregate_survives_all_nan_csdr():
    """整批 cSDR 都是 NaN 时，聚合不应崩溃，而应只保留还算得出来的指标。"""
    refs, mixture = _toy_song(seconds=1.0)
    with pytest.warns(RuntimeWarning):
        s = evaluate_track(refs, silence_baseline(mixture), "x")
    agg = aggregate([s])
    assert "uSDR" in agg
    assert "cSDR" not in agg  # 全 NaN 的指标被整体丢弃，不会污染结果表


def test_evaluate_track_requires_all_stems():
    refs, mixture = _toy_song(seconds=1.0)
    with pytest.raises(KeyError):
        evaluate_track(refs, {"vocals": mixture}, compute_csdr=False)


def test_evaluate_track_tolerates_length_mismatch():
    """分块推理常让输出长几百个点，评测应当自动对齐而不是崩掉。"""
    refs, mixture = _toy_song(seconds=1.0)
    ests = {k: np.concatenate([v, np.zeros((512, 2), np.float32)]) for k, v in refs.items()}
    scores = evaluate_track(refs, ests, compute_csdr=False)
    assert scores.usdr["vocals"] > 100.0


# --------------------------------------------------------------------------------------
# IRM oracle：上界锚点
# --------------------------------------------------------------------------------------

def test_irm_oracle_beats_trivial_baseline():
    """**评测框架的整体自检**：区间的两端都算得对。

    Note:
        原来这里写的是"真实模型的成绩必须落在 [平凡基线, IRM oracle] 之间"。
        **M1 证明这句话只对掩码类方法成立。** Demucs 直接生成波形、能修正相位，
        不受掩码天花板约束 —— 实测它在 drums/bass 上已与 IRM oracle 统计上不可区分。
        所以 IRM 只是"掩码方法的上界"，不是通用上界。
    """
    refs, mixture = _toy_song(seconds=3.0)

    oracle = ideal_ratio_mask(refs, mixture)
    trivial = trivial_baseline(mixture)

    for stem in refs:
        n = refs[stem].shape[0]
        o = global_sdr(refs[stem], match_length(oracle[stem], n))
        t = global_sdr(refs[stem], match_length(trivial[stem], n))
        assert o > t + 5.0, f"{stem}: oracle {o:.1f} dB 没有明显优于平凡基线 {t:.1f} dB"


def test_oracle_sir_and_sar_are_sane():
    """**回归测试，钉住 2026-07-29 那个坑。**

    第一版合成数据用纯正弦 + ``np.roll`` 做立体声，导致 BSS Eval 的
    512 阶 FIR 投影子空间退化，SIR 变成 -114 dB、SAR 恒为 0.00 dB
    （SDR/ISR 却看着正常，极具迷惑性）。

    对一个理想掩码分离，正确的形态应当是：**SIR > SDR**（干扰被压得比总失真更干净），
    且 SAR、ISR 都是正常的正值。
    """
    refs, mixture = _toy_song(seconds=3.0)
    est = {k: match_length(v, mixture.shape[0]) for k, v in ideal_ratio_mask(refs, mixture).items()}
    s = evaluate_track(refs, est, "oracle")

    for stem in refs:
        assert 0.0 < s.csar[stem] < 200.0, f"{stem} SAR={s.csar[stem]:.1f} dB 不合理"
        assert 0.0 < s.cisr[stem] < 200.0, f"{stem} ISR={s.cisr[stem]:.1f} dB 不合理"
        assert s.csir[stem] > s.csdr[stem], (
            f"{stem}: SIR({s.csir[stem]:.1f}) 应当高于 SDR({s.csdr[stem]:.1f})"
        )


def test_irm_oracle_preserves_shape():
    refs, mixture = _toy_song(seconds=1.0)
    oracle = ideal_ratio_mask(refs, mixture)
    for stem in refs:
        assert oracle[stem].shape == mixture.shape


def test_irm_beats_ibm():
    """IRM（软掩码）必须优于 IBM（硬掩码）。

    这是掩码实现的**交叉验证**：如果硬掩码反而更好，说明比值掩码或 iSTFT 写错了。
    M1 在真实数据上量到的差距是 9.23 vs 8.81 dB。
    """
    refs, mixture = _toy_song(seconds=3.0)
    n = mixture.shape[0]
    irm = ideal_ratio_mask(refs, mixture)
    ibm = ideal_binary_mask(refs, mixture)

    irm_mean = np.mean([global_sdr(refs[s], match_length(irm[s], n)) for s in refs])
    ibm_mean = np.mean([global_sdr(refs[s], match_length(ibm[s], n)) for s in refs])
    assert irm_mean > ibm_mean, f"IRM {irm_mean:.2f} 竟不如 IBM {ibm_mean:.2f} dB"


def test_ibm_mask_is_a_partition():
    """IBM 是硬划分：四轨估计之和必须精确等于混音（每个时频点只归一个声部）。

    这一条能抓到"胜者判定写错导致某些时频点被重复计入或漏掉"的错误。
    """
    refs, mixture = _toy_song(seconds=2.0)
    ibm = ideal_binary_mask(refs, mixture)
    total = sum(ibm.values())
    n = min(mixture.shape[0], total.shape[0])
    # iSTFT 的边界效应会带来极小误差，用相对量判断
    err = np.max(np.abs(mixture[:n] - total[:n])) / np.max(np.abs(mixture))
    assert err < 0.02, f"IBM 四轨之和与混音的相对偏差 {err:.4f} 过大，掩码不是硬划分"


# --------------------------------------------------------------------------------------
# 聚合与输出
# --------------------------------------------------------------------------------------

def _fake_track(name: str, values: dict[str, float]) -> TrackScores:
    return TrackScores(track_name=name, csdr=dict(values), usdr=dict(values))


def test_aggregate_median_and_mean():
    tracks = [
        _fake_track("a", {"vocals": 6.0, "drums": 5.0, "bass": 4.0, "other": 3.0}),
        _fake_track("b", {"vocals": 8.0, "drums": 7.0, "bass": 6.0, "other": 5.0}),
        _fake_track("c", {"vocals": 10.0, "drums": 9.0, "bass": 8.0, "other": 7.0}),
    ]
    med = aggregate(tracks, agg="median")["cSDR"]
    assert med["vocals"] == pytest.approx(8.0)
    # "平均"列是四轨的算术平均，不是再取一次中位数
    assert med["mean"] == pytest.approx((8.0 + 7.0 + 6.0 + 5.0) / 4)

    avg = aggregate(tracks, agg="mean")["cSDR"]
    assert avg["vocals"] == pytest.approx(8.0)


def test_aggregate_ignores_non_finite():
    tracks = [
        _fake_track("a", {"vocals": 6.0, "drums": 5.0, "bass": 4.0, "other": 3.0}),
        _fake_track("b", {"vocals": float("nan"), "drums": 7.0, "bass": 6.0, "other": 5.0}),
    ]
    med = aggregate(tracks, agg="mean")["cSDR"]
    assert med["vocals"] == pytest.approx(6.0)


def test_per_stem_matrix_shape():
    tracks = [_fake_track(str(i), dict.fromkeys(STEMS, float(i))) for i in range(5)]
    m = per_stem_matrix(tracks)
    assert m.shape == (5, 4)
    assert m[3, 0] == pytest.approx(3.0)


def test_format_table_renders_markdown():
    rows = {"htdemucs": {"vocals": 8.1, "drums": 8.5, "bass": 8.6, "other": 5.5, "mean": 7.68}}
    table = format_table(rows)
    assert "| htdemucs |" in table
    assert "**7.68**" in table
