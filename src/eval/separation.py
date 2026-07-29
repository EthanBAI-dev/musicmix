"""音源分离评测。

**这个文件是整个项目可信度的地基。** 三条纪律：

1. **cSDR 用官方 museval，不自己实现。** BSS Eval v4 里那套 512 阶失真滤波器
   自己写十有八九和论文不可比。我们只负责正确地调用与聚合。
2. **cSDR 与 uSDR 是两套不同的东西，数值差 1~2 dB，报数必须写明口径。**
   - ``cSDR``：museval / BSS Eval v4，1 秒一块，块内算 SDR，**取中位数**。
     SiSEC、MUSDB18 论文用的是这个。
   - ``uSDR``：整首歌一个值的 "global SDR"，Music Demixing Challenge 用的是这个。
3. **评测前不要做响度归一化。** SDR 是幅度敏感的。

参考：
- BSS Eval v4 / museval: https://github.com/sigsep/sigsep-mus-eval
- MDX Challenge 的 global SDR 定义（本文件 :func:`global_sdr` 与之逐字对齐）
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np

STEMS = ("vocals", "drums", "bass", "other")

# MDX Challenge 官方 global SDR 实现里用的常数，照抄以保证可比
_MDX_DELTA = 1e-7


# --------------------------------------------------------------------------------------
# 逐首歌的指标
# --------------------------------------------------------------------------------------

def global_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """整首歌一个值的 SDR，即 Music Demixing Challenge 的 "uSDR"。

    .. math:: \\mathrm{SDR} = 10\\log_{10}\\frac{\\sum r^2 + \\delta}{\\sum (r-e)^2 + \\delta}

    求和跨越全部采样点与全部声道。

    Args:
        reference: 真值 ``(n_samples, n_channels)``
        estimate: 估计 ``(n_samples, n_channels)``

    Returns:
        SDR，单位 dB。

    Note:
        一个有用的心算锚点：**估计全零时，SDR 恰好等于 0 dB**（分子分母相等）。
        单测就是拿这个当断言的。所以"SDR 大于 0"只说明比"输出静音"强，
        并不说明分离得好。
    """
    reference = np.asarray(reference, dtype=np.float64)
    estimate = np.asarray(estimate, dtype=np.float64)
    if reference.shape != estimate.shape:
        raise ValueError(f"形状不一致：ref={reference.shape} est={estimate.shape}")
    num = np.sum(reference**2) + _MDX_DELTA
    den = np.sum((reference - estimate) ** 2) + _MDX_DELTA
    return float(10.0 * np.log10(num / den))


def si_sdr(reference: np.ndarray, estimate: np.ndarray, zero_mean: bool = True) -> float:
    """尺度不变 SDR（Le Roux et al. 2019）。

    先把 estimate 里与 reference 共线的部分投影出来当作目标，剩下的算噪声。
    因此对整体音量缩放免疫 —— ``si_sdr(r, 2*r) == si_sdr(r, r) == +inf``。

    本项目只把它当辅助指标（分离模型有时会有系统性的增益偏差，
    SI-SDR 能把"音量不对"和"内容不对"区分开）。主指标仍是 museval 的 cSDR。
    """
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(estimate, dtype=np.float64).reshape(-1)
    if reference.shape != estimate.shape:
        raise ValueError(f"形状不一致：ref={reference.shape} est={estimate.shape}")
    if zero_mean:
        reference = reference - reference.mean()
        estimate = estimate - estimate.mean()

    ref_energy = np.sum(reference**2)
    if ref_energy == 0:
        return float("nan")
    alpha = np.sum(reference * estimate) / ref_energy
    target = alpha * reference
    noise = estimate - target
    noise_energy = np.sum(noise**2)
    target_energy = np.sum(target**2)

    if noise_energy == 0:
        # 估计全零时 target 和 noise 同时为 0。这不是"完美"，是"什么都没输出"，
        # 属于最差情况，必须给 -inf。早期版本这里直接返回 +inf，
        # 导致 silence 基线的 SI-SDR 变成 +∞ —— 2026-07-29 的 M0 自检抓到的真 bug。
        return float("inf") if target_energy > 0 else float("-inf")
    if target_energy == 0:
        return float("-inf")
    return float(10.0 * np.log10(target_energy / noise_energy))


@dataclass
class TrackScores:
    """一首歌的分离评测结果。

    ``csdr`` 等字段是 ``{stem: value}``；``csdr_frames`` 保留逐块原始值，
    做 bootstrap 和失败案例定位时要用。
    """

    track_name: str
    csdr: dict[str, float] = field(default_factory=dict)
    cisr: dict[str, float] = field(default_factory=dict)
    csir: dict[str, float] = field(default_factory=dict)
    csar: dict[str, float] = field(default_factory=dict)
    usdr: dict[str, float] = field(default_factory=dict)
    sisdr: dict[str, float] = field(default_factory=dict)
    csdr_frames: dict[str, np.ndarray] = field(default_factory=dict)

    def mean_csdr(self) -> float:
        vals = [v for v in self.csdr.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    def to_row(self) -> dict[str, float | str]:
        row: dict[str, float | str] = {"track": self.track_name}
        for stem, v in self.csdr.items():
            row[f"cSDR_{stem}"] = v
        row["cSDR_mean"] = self.mean_csdr()
        for stem, v in self.usdr.items():
            row[f"uSDR_{stem}"] = v
        return row


def evaluate_track(
    references: dict[str, np.ndarray],
    estimates: dict[str, np.ndarray],
    track_name: str = "unnamed",
    sr: int = 44100,
    win_seconds: float = 1.0,
    hop_seconds: float = 1.0,
    compute_csdr: bool = True,
) -> TrackScores:
    """评一首歌的四轨分离结果。

    Args:
        references: ``{stem: (n, ch)}`` 真值。
        estimates: ``{stem: (n, ch)}`` 估计，必须含有 references 的全部 key。
        win_seconds/hop_seconds: museval 的分块窗长与跳步，**默认 1 秒是官方口径，别改**。
        compute_csdr: 关掉可以跳过 museval（很慢），只算 uSDR/SI-SDR，用于快速迭代。

    Returns:
        :class:`TrackScores`
    """
    stems = list(references.keys())
    missing = [s for s in stems if s not in estimates]
    if missing:
        raise KeyError(f"估计里缺少声部：{missing}")

    # 长度对齐：分块推理常让输出比输入长几百个点
    n = min(min(references[s].shape[0] for s in stems), min(estimates[s].shape[0] for s in stems))
    ref_arr = np.stack([np.asarray(references[s])[:n] for s in stems], axis=0)
    est_arr = np.stack([np.asarray(estimates[s])[:n] for s in stems], axis=0)

    scores = TrackScores(track_name=track_name)

    for i, stem in enumerate(stems):
        scores.usdr[stem] = global_sdr(ref_arr[i], est_arr[i])
        scores.sisdr[stem] = si_sdr(ref_arr[i], est_arr[i])

    if compute_csdr:
        import museval

        try:
            sdr, isr, sir, sar = museval.evaluate(
                ref_arr,
                est_arr,
                win=int(win_seconds * sr),
                hop=int(hop_seconds * sr),
                mode="v4",
                padding=True,
            )
        except ValueError as e:
            # BSS Eval v4 要求每一路参考和估计都非全零，否则投影方程组欠定。
            # 真实场景会遇到：某些歌本来就没有人声轨，或模型对某一轨输出了静音。
            # 这里给 NaN 并**明确警告**（aggregate 会跳过 NaN），绝不静默吞掉。
            warnings.warn(
                f"[{track_name}] museval 无法计算 cSDR：{e}\n"
                f"  → 该曲 cSDR 记为 NaN，uSDR/SI-SDR 不受影响。"
                f"  如果大量曲目触发这条，说明模型在整轨输出静音，是真问题。",
                RuntimeWarning,
                stacklevel=2,
            )
            for stem in stems:
                scores.csdr[stem] = float("nan")
            return scores

        for i, stem in enumerate(stems):
            scores.csdr_frames[stem] = np.asarray(sdr[i], dtype=np.float64)
            # 官方聚合：块内取中位数，忽略 NaN（静音块 BSS Eval 会给 NaN）
            scores.csdr[stem] = float(np.nanmedian(sdr[i]))
            scores.cisr[stem] = float(np.nanmedian(isr[i]))
            scores.csir[stem] = float(np.nanmedian(sir[i]))
            scores.csar[stem] = float(np.nanmedian(sar[i]))

    return scores


# --------------------------------------------------------------------------------------
# 跨歌曲聚合
# --------------------------------------------------------------------------------------

def aggregate(
    tracks: list[TrackScores],
    stems: tuple[str, ...] = STEMS,
    agg: str = "median",
) -> dict[str, dict[str, float]]:
    """把逐首结果聚合成一张表。

    Args:
        agg: ``"median"``（museval / MUSDB 官方口径）或 ``"mean"``。
            **cSDR 报中位数，uSDR 报均值**，这是两个社区各自的惯例；
            本函数只按 agg 统一处理，调用方负责在结果表里写清楚。

    Returns:
        ``{metric: {stem: value, "mean": value}}``，metric ∈ {cSDR, uSDR, SI-SDR, ISR, SIR, SAR}
    """
    if not tracks:
        raise ValueError("没有任何评测结果可聚合")
    fn = np.nanmedian if agg == "median" else np.nanmean

    sources = {
        "cSDR": lambda t: t.csdr,
        "uSDR": lambda t: t.usdr,
        "SI-SDR": lambda t: t.sisdr,
        "ISR": lambda t: t.cisr,
        "SIR": lambda t: t.csir,
        "SAR": lambda t: t.csar,
    }

    out: dict[str, dict[str, float]] = {}
    for metric, getter in sources.items():
        per_stem: dict[str, float] = {}
        for stem in stems:
            vals = [getter(t)[stem] for t in tracks if stem in getter(t)]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                per_stem[stem] = float(fn(vals))
        if per_stem:
            # "平均"这一列是四轨的算术平均，不是再取一次中位数
            per_stem["mean"] = float(np.mean(list(per_stem.values())))
            out[metric] = per_stem
    return out


def per_stem_matrix(tracks: list[TrackScores], stems: tuple[str, ...] = STEMS) -> np.ndarray:
    """返回 ``(n_tracks, n_stems)`` 的 cSDR 矩阵，供显著性检验使用。"""
    return np.array(
        [[t.csdr.get(s, np.nan) for s in stems] for t in tracks],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------------------
# 结果表输出
# --------------------------------------------------------------------------------------

def format_table(
    rows: dict[str, dict[str, float]],
    stems: tuple[str, ...] = STEMS,
    metric: str = "cSDR",
    float_fmt: str = "{:.2f}",
) -> str:
    """把 ``{模型名: aggregate() 的某个 metric 字典}`` 渲染成 markdown 表。

    直接贴进 ``results/*.md``。
    """
    header = f"| 模型 | {' | '.join(stems)} | 平均 |"
    sep = "|" + "---|" * (len(stems) + 2)
    lines = [f"**指标：{metric}（dB，越高越好）**", "", header, sep]
    for name, per_stem in rows.items():
        cells = [float_fmt.format(per_stem.get(s, float("nan"))) for s in stems]
        mean = float_fmt.format(per_stem.get("mean", float("nan")))
        lines.append(f"| {name} | {' | '.join(cells)} | **{mean}** |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 参照基线：这两个函数存在的意义是"验证评测本身没写错"
# --------------------------------------------------------------------------------------

def trivial_baseline(mixture: np.ndarray, stems: tuple[str, ...] = STEMS) -> dict[str, np.ndarray]:
    """平凡基线：把混音原样当作每一个声部的估计。

    这是**下界锚点**。它的 uSDR 必然接近但通常低于 0 dB，
    cSDR 通常在 0 dB 上下。如果评测代码算出一个很高的分，说明评测写错了。
    """
    return {s: np.array(mixture, copy=True) for s in stems}


def silence_baseline(mixture: np.ndarray, stems: tuple[str, ...] = STEMS) -> dict[str, np.ndarray]:
    """静音基线：全部输出零。

    **uSDR 必然恰好等于 0.00 dB**（分子分母相等）。这是最硬的自检断言：
    如果这一条不成立，:func:`global_sdr` 一定实现错了。
    """
    return {s: np.zeros_like(mixture) for s in stems}


def ideal_ratio_mask(
    references: dict[str, np.ndarray],
    mixture: np.ndarray,
    n_fft: int = 4096,
    hop_length: int = 1024,
    power: float = 2.0,
) -> dict[str, np.ndarray]:
    """IRM oracle：用真值算出的理想比值掩码，作为**上界锚点**。

    含义："在'预测频谱掩码再 iSTFT'这个框架下，最好能到多少 dB"。
    真实模型的 SDR 应该落在 :func:`trivial_baseline` 和这个之间；
    落到外面就是哪里错了。

    Args:
        power: 1.0 = 幅度掩码，2.0 = 功率掩码（更常用）。

    Returns:
        ``{stem: (n, ch)}``，长度与 mixture 对齐。
    """
    import librosa

    from src.audio.io import match_length

    stems = list(references.keys())
    n_samples, n_ch = mixture.shape

    # 逐声道做 STFT：librosa 对 (…, n) 支持批处理，这里显式循环更好读
    specs: dict[str, list[np.ndarray]] = {s: [] for s in stems}
    mix_specs: list[np.ndarray] = []
    for c in range(n_ch):
        mix_specs.append(librosa.stft(mixture[:, c], n_fft=n_fft, hop_length=hop_length))
        for s in stems:
            ref_c = match_length(references[s], n_samples)[:, c]
            specs[s].append(librosa.stft(ref_c, n_fft=n_fft, hop_length=hop_length))

    out: dict[str, np.ndarray] = {}
    eps = 1e-10
    denom = [
        sum(np.abs(specs[s][c]) ** power for s in stems) + eps for c in range(n_ch)
    ]
    for s in stems:
        chans = []
        for c in range(n_ch):
            mask = (np.abs(specs[s][c]) ** power) / denom[c]
            y = librosa.istft(mask * mix_specs[c], hop_length=hop_length, length=n_samples)
            chans.append(y)
        out[s] = np.stack(chans, axis=1).astype(np.float32)
    return out
