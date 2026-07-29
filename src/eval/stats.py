"""显著性检验。

**为什么需要这个文件：** MUSDB18-HQ 测试集只有 50 首歌。在这个样本量下，
两个方法平均 SDR 差 0.1 dB 很可能只是噪声。只报均值就宣称"提升了"是站不住的。

本项目的规矩（写进 [03-评测协议]）：**任何声称的改进都要附配对检验的置信区间。**
因为是同样的 50 首歌跑两个方法，属于**配对样本**，配对检验的功效远高于独立样本检验。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PairedResult:
    """配对比较结果。"""

    mean_diff: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int
    method: str

    @property
    def significant(self) -> bool:
        """置信区间不跨 0 即认为显著。"""
        return (self.ci_low > 0) or (self.ci_high < 0)

    def describe(self, unit: str = "dB") -> str:
        star = "显著" if self.significant else "不显著"
        return (
            f"Δ = {self.mean_diff:+.3f} {unit} "
            f"[95% CI: {self.ci_low:+.3f}, {self.ci_high:+.3f}], "
            f"p = {self.p_value:.4f}, n = {self.n} → **{star}**"
        )


def paired_bootstrap(
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedResult:
    """配对 bootstrap：对"歌"重采样，估计平均提升的置信区间。

    做法：把 50 首歌有放回地重抽 50 首，每次算两法均值之差，重复 n_boot 次，
    取差值分布的 2.5% / 97.5% 分位数作为 95% CI。

    Args:
        baseline: ``(n,)`` 基线方法在每首歌上的指标。
        treatment: ``(n,)`` 新方法在**同样这些歌**上的指标，顺序必须一一对应。

    Returns:
        :class:`PairedResult`，``mean_diff = mean(treatment) - mean(baseline)``。
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(treatment, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"配对样本长度不一致：{a.shape} vs {b.shape}")

    diff = b - a
    finite = np.isfinite(diff)
    diff = diff[finite]
    n = diff.size
    if n < 2:
        raise ValueError("有效配对样本不足 2 个")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)

    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # 双侧 p：bootstrap 分布落在 0 另一侧的比例 ×2
    p = 2 * min((boot_means <= 0).mean(), (boot_means >= 0).mean())
    return PairedResult(
        mean_diff=float(diff.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=float(min(p, 1.0)),
        n=int(n),
        method="paired bootstrap",
    )


def paired_permutation(
    baseline: np.ndarray,
    treatment: np.ndarray,
    n_perm: int = 10_000,
    seed: int = 0,
) -> PairedResult:
    """配对置换检验：随机翻转每首歌上"谁是基线"的符号，得到零假设分布。

    比 bootstrap 更适合出 p 值（bootstrap 更适合出 CI），两个一起报最稳妥。
    CI 字段这里复用 bootstrap 的结果以免误导，只用它的 p_value。
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(treatment, dtype=np.float64)
    diff = (b - a)[np.isfinite(b - a)]
    n = diff.size
    if n < 2:
        raise ValueError("有效配对样本不足 2 个")

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    null_means = (signs * diff[None, :]).mean(axis=1)
    observed = diff.mean()
    p = float((np.abs(null_means) >= abs(observed)).mean())

    boot = paired_bootstrap(a, b, seed=seed)
    return PairedResult(
        mean_diff=float(observed),
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        p_value=p,
        n=int(n),
        method="paired permutation (CI from bootstrap)",
    )


def compare_models(
    baseline_per_track: np.ndarray,
    treatment_per_track: np.ndarray,
    stems: tuple[str, ...] = ("vocals", "drums", "bass", "other"),
    seed: int = 0,
) -> dict[str, PairedResult]:
    """逐声部 + 平均，一次性给出全部配对检验结果。

    Args:
        baseline_per_track / treatment_per_track: ``(n_tracks, n_stems)``，
            即 :func:`src.eval.separation.per_stem_matrix` 的输出。

    Returns:
        ``{"vocals": PairedResult, ..., "mean": PairedResult}``
    """
    a = np.asarray(baseline_per_track, dtype=np.float64)
    b = np.asarray(treatment_per_track, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"形状不一致：{a.shape} vs {b.shape}")

    out = {stem: paired_bootstrap(a[:, i], b[:, i], seed=seed) for i, stem in enumerate(stems)}
    out["mean"] = paired_bootstrap(np.nanmean(a, axis=1), np.nanmean(b, axis=1), seed=seed)
    return out


def format_comparison(results: dict[str, PairedResult], unit: str = "dB") -> str:
    """渲染成 markdown，直接贴进 ``results/*.md``。"""
    lines = [
        "| 声部 | Δ | 95% CI | p | 显著 |",
        "|---|---|---|---|---|",
    ]
    for stem, r in results.items():
        lines.append(
            f"| {stem} | {r.mean_diff:+.3f} {unit} | "
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.p_value:.4f} | "
            f"{'✅' if r.significant else '❌'} |"
        )
    return "\n".join(lines)
