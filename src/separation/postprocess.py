"""推理期增益：不重训模型，只在推理和后处理上做文章（P2 路线 A）。

三种手段，各自独立可开关，方便做**逐项累加的消融表**：

1. **软掩码细化（α-Wiener）** —— 把模型输出的幅度当作掩码依据重新分配混音的复数谱。
   副作用是强制满足 ``Σ estimates == mixture``。
2. **多通道维纳滤波（MWF）** —— 在软掩码基础上再考虑各声部的**空间协方差**，
   对立体声形象有帮助。计算量大一些。
3. **测试时增强（TTA）** —— 对输入做等价变换、分别推理、再变换回来求平均。
   这里用**声道交换**和**极性翻转**：两者都是理论上的等价变换（真实分离结果应当同变），
   所以平均只会抵消噪声、不会引入偏差。

.. note::
   **P1 给出的可证伪预测**（记在这里，跑完拿实测对照）：

   掩码类后处理只能重新分配混音里已有的复数谱，天花板就是 IRM oracle。
   P1 实测 htdemucs 在 **drums / bass 上已与 IRM oracle 统计上不可区分**
   （配对 bootstrap p=0.88 / 0.18），而 **other 差 2.59 dB、vocals 差 2.31 dB 且显著**。

   → 所以预测：**软掩码/MWF 在 other 和 vocals 上应当有增益，在 drums 和 bass 上不应当有。**
   如果 drums/bass 也涨了，说明我对"已触及天花板"的判断有问题，要回头查。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

N_FFT = 4096
HOP = 1024
EPS = 1e-10
# MWF 的频段分块大小。见 multichannel_wiener 里的说明：
# 不分块的话单个中间数组对一首 4 分钟的歌就有 677 MB。
FREQ_BLOCK = 256


# --------------------------------------------------------------------------------------
# STFT 辅助
# --------------------------------------------------------------------------------------

def _stft_multi(x: np.ndarray, n_fft: int = N_FFT, hop: int = HOP) -> np.ndarray:
    """``(n, ch)`` → ``(ch, freq, frames)`` 复数谱。"""
    import librosa

    return np.stack(
        [librosa.stft(np.ascontiguousarray(x[:, c]), n_fft=n_fft, hop_length=hop)
         for c in range(x.shape[1])],
        axis=0,
    )


def _istft_multi(X: np.ndarray, length: int, hop: int = HOP) -> np.ndarray:
    """``(ch, freq, frames)`` → ``(n, ch)``。"""
    import librosa

    return np.stack(
        [librosa.istft(X[c], hop_length=hop, length=length) for c in range(X.shape[0])],
        axis=1,
    ).astype(np.float32)


# --------------------------------------------------------------------------------------
# 1. 软掩码细化（α-Wiener）
# --------------------------------------------------------------------------------------

def soft_mask_refine(
    estimates: dict[str, np.ndarray],
    mixture: np.ndarray,
    alpha: float = 2.0,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> dict[str, np.ndarray]:
    """用模型输出的幅度构造软掩码，重新分配**混音**的复数谱。

    .. math:: M_j = \\frac{|\\hat S_j|^\\alpha}{\\sum_k |\\hat S_k|^\\alpha},
              \\qquad \\tilde S_j = M_j \\cdot X

    这和 :func:`src.eval.separation.ideal_ratio_mask` 是同一个公式，
    只是把真值幅度换成了模型估计 —— 所以它的天花板就是 IRM oracle。

    Args:
        alpha: 1.0 幅度掩码，2.0 功率掩码（更常用也更强）。

    Returns:
        细化后的估计。**恒满足 Σ estimates == mixture**（掩码逐点求和为 1）。
    """
    stems = list(estimates.keys())
    n, n_ch = mixture.shape

    X = _stft_multi(mixture, n_fft, hop)
    S = {s: _stft_multi(_fit(estimates[s], n), n_fft, hop) for s in stems}

    mags = {s: np.abs(S[s]) ** alpha for s in stems}
    denom = sum(mags.values()) + EPS

    return {s: _istft_multi(mags[s] / denom * X, n, hop) for s in stems}


# --------------------------------------------------------------------------------------
# 2. 多通道维纳滤波
# --------------------------------------------------------------------------------------

def multichannel_wiener(
    estimates: dict[str, np.ndarray],
    mixture: np.ndarray,
    n_iter: int = 1,
    alpha: float = 2.0,
    n_fft: int = N_FFT,
    hop: int = HOP,
) -> dict[str, np.ndarray]:
    """多通道维纳滤波（norbert 风格的单/多轮 EM）。

    在软掩码之上多做一件事：为每个声部估计一个 ``(ch, ch)`` 的**空间协方差** ``R_j(f)``，
    于是滤波器变成

    .. math:: \\tilde S_j = v_j R_j \\left(\\sum_k v_k R_k\\right)^{-1} X

    其中 ``v_j(t,f)`` 是功率谱密度。声道间相关性被建模进来，
    对立体声形象（ISR 指标）通常有帮助。

    Args:
        n_iter: EM 轮数。1 轮通常就够，>1 收益很小但耗时线性增长。

    Note:
        这里只对 2 声道做闭式求逆（音乐分离固定立体声），
        所以下面直接写了 2×2 的伴随矩阵求逆，比通用 ``np.linalg.inv`` 快很多，
        也避免了在 ``(freq, frames)`` 上循环。
    """
    stems = list(estimates.keys())
    n, n_ch = mixture.shape
    if n_ch != 2:
        raise ValueError(f"MWF 这里只实现了立体声，得到 {n_ch} 声道")

    X = _stft_multi(mixture, n_fft, hop)                      # (2, F, T)
    S = {s: _stft_multi(_fit(estimates[s], n), n_fft, hop) for s in stems}

    n_freq = X.shape[1]

    for _ in range(n_iter):
        # 功率谱密度：声道平均
        v = {s: np.mean(np.abs(S[s]) ** 2, axis=0) + EPS for s in stems}       # (F, T)

        # 空间协方差 R_j(f)：对时间求和后归一化，得到 (2, 2, F)
        R = {}
        for s in stems:
            Sj = S[s]
            num = np.einsum("aft,bft->abf", Sj, np.conj(Sj))
            R[s] = num / (np.sum(v[s], axis=-1)[None, None, :] + EPS)

        # **按频段分块处理。** 中间量 C / inv / W 都是 (2,2,F,T)：
        # 一首 4 分钟的歌 F≈2049、T≈10336，单个数组就有 677 MB（complex64），
        # 多开几个 worker 直接把内存打满。分块后峰值只和 FREQ_BLOCK 成正比。
        new_S = {s: np.empty_like(S[s]) for s in stems}
        for f0 in range(0, n_freq, FREQ_BLOCK):
            f1 = min(f0 + FREQ_BLOCK, n_freq)
            Xb = X[:, f0:f1, :]
            vb = {s: v[s][f0:f1, :] for s in stems}
            Rb = {s: R[s][:, :, f0:f1] for s in stems}

            # 混合协方差 Σ_k v_k R_k → (2, 2, f, T)
            C = sum(vb[s][None, None] * Rb[s][:, :, :, None] for s in stems)

            # 2×2 闭式求逆（伴随矩阵 / 行列式），比通用 inv 快得多
            a, b, c, d = C[0, 0], C[0, 1], C[1, 0], C[1, 1]
            det = a * d - b * c
            det = np.where(np.abs(det) < EPS, EPS, det)
            inv = np.stack([np.stack([d, -b]), np.stack([-c, a])]) / det

            GX = np.einsum("abft,bft->aft", inv, Xb)
            for s in stems:
                W = vb[s][None, None] * Rb[s][:, :, :, None]
                new_S[s][:, f0:f1, :] = np.einsum("abft,bft->aft", W, GX)
        S = new_S

    return {s: _istft_multi(S[s], n, hop) for s in stems}


# --------------------------------------------------------------------------------------
# 3. 测试时增强
# --------------------------------------------------------------------------------------

# 每个变换都是自逆的（做两次回到原样），所以正变换和逆变换共用一个函数
TTA_TRANSFORMS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "swap": lambda x: x[:, ::-1].copy(),          # 左右声道交换
    "flip": lambda x: -x,                          # 极性翻转
    "swap_flip": lambda x: -x[:, ::-1],
}


def apply_tta(
    separate_fn: Callable[[np.ndarray], dict[str, np.ndarray]],
    mixture: np.ndarray,
    variants: tuple[str, ...] = ("identity", "swap", "flip"),
) -> dict[str, np.ndarray]:
    """对若干等价变换分别推理，逆变换回来后在**时域**求平均。

    为什么这几个变换是"等价"的：真实的分离结果对左右交换和极性翻转是**同变**的
    （交换输入声道，输出声道也该交换；输入取反，输出也该取反）。
    所以逆变换后它们估计的是同一个目标，平均只抵消模型噪声、不引入偏差。

    Args:
        separate_fn: ``mixture -> {stem: (n,2)}``，通常是 demucs 推理的偏函数。
        variants: :data:`TTA_TRANSFORMS` 的键。**推理次数 = len(variants)**，
            RTF 线性增长，消融表里必须把这个代价一起报。
    """
    acc: dict[str, np.ndarray] = {}
    n = mixture.shape[0]

    for name in variants:
        t = TTA_TRANSFORMS[name]
        out = separate_fn(t(np.ascontiguousarray(mixture)))
        for stem, y in out.items():
            y = t(_fit(y, n))          # 变换自逆，直接再作用一次即还原
            acc[stem] = y.astype(np.float64) if stem not in acc else acc[stem] + y

    k = len(variants)
    return {s: (v / k).astype(np.float32) for s, v in acc.items()}


# --------------------------------------------------------------------------------------
# 4. 多模型集成
# --------------------------------------------------------------------------------------

def ensemble(
    outputs: list[dict[str, np.ndarray]],
    weights: dict[str, list[float]] | list[float] | None = None,
) -> dict[str, np.ndarray]:
    """多个模型输出的加权平均（时域）。

    Args:
        outputs: 每个模型的 ``{stem: (n,2)}``。
        weights: 统一权重列表，或 ``{stem: [w1, w2, ...]}`` **逐声部**权重。
            逐声部调权是有道理的 —— 不同模型的强项不同
            （P1 实测 htdemucs 的 drums 强、other 弱）。None 表示等权。
    """
    if not outputs:
        raise ValueError("没有可集成的输出")
    stems = list(outputs[0].keys())
    n = min(min(y.shape[0] for y in o.values()) for o in outputs)

    out = {}
    for s in stems:
        if isinstance(weights, dict):
            w = np.asarray(weights.get(s, [1.0] * len(outputs)), dtype=np.float64)
        elif weights is None:
            w = np.ones(len(outputs))
        else:
            w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        out[s] = sum(w[i] * o[s][:n].astype(np.float64) for i, o in enumerate(outputs)).astype(np.float32)
    return out


# --------------------------------------------------------------------------------------

def _fit(x: np.ndarray, n: int) -> np.ndarray:
    """长度对齐到 n（截断或补零）。"""
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - x.shape[0], x.shape[1]), dtype=x.dtype)])


def sum_error(mixture: np.ndarray, estimates: dict[str, np.ndarray]) -> float:
    """``max|mixture - Σestimates|``。

    软掩码细化之后这个值应当接近 0（掩码逐点和为 1）；
    Demucs 原始输出则不然（它不是掩码方法）。可作为"细化是否真的生效"的自检。
    """
    total = sum(estimates.values())
    n = min(mixture.shape[0], total.shape[0])
    return float(np.max(np.abs(mixture[:n] - total[:n])))
