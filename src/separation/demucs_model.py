"""Demucs 推理封装（P1 baseline）。

**这个文件的唯一目标是「精确复刻官方推理流程」**，而不是写得好看。
任何自作聪明的改动都会让 SDR 对不上论文，然后你要花半天排查。

必须照抄的三件事：

1. **用混音自身的 mean/std 做归一化，推理后再还原。**
   这是 Demucs 官方 ``separate.py`` 的做法，不是可选项 —— 少了它 SDR 会掉。
2. **分块推理的重叠率默认 0.25。** 官方默认值。改大能涨一点 SDR 但更慢，
   属于 P2-A 的"推理期增益"路线，要在消融表里单独列，**不能偷偷改了当基线**。
3. **声部顺序按 ``model.sources`` 取，不要写死。**
   htdemucs 的顺序是 ``['drums','bass','other','vocals']``，和我们的
   ``('vocals','drums','bass','other')`` 不一样。写死必然错位，
   而错位的表现是"SDR 很低但不报错"，极难发现。

参考：https://github.com/adefossez/demucs
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# 官方权重名 → 说明
AVAILABLE = {
    "htdemucs": "Hybrid Transformer Demucs v4，论文报 MUSDB18 平均 SDR ≈ 9.0 dB",
    "htdemucs_ft": "htdemucs 的逐声部微调版，≈ 9.20 dB，慢 4 倍（4 个模型各跑一遍）",
    "htdemucs_6s": "6 声部版（多出 piano / guitar），4 轨指标不可与上面直接比",
    "mdx_extra": "MDX 挑战赛版，时域 Demucs",
    "mdx_extra_q": "mdx_extra 的量化版，小而快",
}


def pick_device(prefer: str = "auto") -> str:
    """选设备。

    Note:
        MPS 上部分算子会**静默回退 CPU**，所以性能数字必须和 CPU 档交叉验证。
        如果 MPS 上跑出来的 SDR 与 CPU 不一致（超过 0.05 dB），说明有算子行为差异，
        以 CPU 结果为准并记进 DEVLOG。
    """
    import torch

    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DemucsSeparator:
    """一个加载好的 Demucs 模型。模型常驻，避免每首歌重新加载权重。"""

    name: str
    device: str
    overlap: float
    shifts: int
    model: object
    sources: tuple[str, ...]

    def __call__(self, mixture: np.ndarray) -> dict[str, np.ndarray]:
        return separate(self, mixture)


def load(
    name: str = "htdemucs",
    device: str = "auto",
    overlap: float = 0.25,
    shifts: int = 0,
) -> DemucsSeparator:
    """加载预训练权重。

    Args:
        name: 见 :data:`AVAILABLE`。
        overlap: 分块推理重叠率，**官方默认 0.25，做 baseline 时不要改**。
        shifts: 随机时移 TTA 的次数，官方默认 0。>0 就属于 P2-A 的改进，
            必须在结果表里注明。
    """
    import torch
    from demucs.pretrained import get_model

    if name not in AVAILABLE:
        warnings.warn(f"{name!r} 不在已知列表里，仍尝试加载：{list(AVAILABLE)}", stacklevel=2)

    dev = pick_device(device)
    model = get_model(name)
    model.eval()
    model.to(dev)

    # 冻结梯度：推理全程不需要，省显存
    for p in model.parameters():
        p.requires_grad_(False)

    sources = tuple(model.sources)
    if shifts > 0 or overlap != 0.25:
        warnings.warn(
            f"非默认推理参数（overlap={overlap}, shifts={shifts}）—— "
            f"这已经不是 baseline，结果表里必须注明。",
            stacklevel=2,
        )
    return DemucsSeparator(
        name=name, device=dev, overlap=overlap, shifts=shifts, model=model, sources=sources
    )


def separate(sep: DemucsSeparator, mixture: np.ndarray) -> dict[str, np.ndarray]:
    """分离一首歌。

    Args:
        sep: :func:`load` 的返回值。
        mixture: ``(n_samples, 2) float32 @ 44.1kHz``（本项目全局约定）。

    Returns:
        ``{stem: (n_samples, 2) float32}``，长度与输入严格一致。
    """
    import torch
    from demucs.apply import apply_model

    x = np.asarray(mixture, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"期望 (n, 2) 立体声，得到 {x.shape}")
    n = x.shape[0]

    # 本项目用 (n, ch)，torch/demucs 用 (ch, n) —— 每次交互都显式转置，不靠记忆
    wav = torch.from_numpy(x.T).contiguous()

    # ---- 官方归一化：用混音的单声道均值算 mean/std ----
    ref = wav.mean(dim=0)
    mean, std = ref.mean(), ref.std()
    if std < 1e-8:                      # 整首静音，直接返回零
        return {s: np.zeros_like(x) for s in sep.sources}
    wav = (wav - mean) / std

    with torch.no_grad():
        out = apply_model(
            sep.model,
            wav[None],                  # (batch=1, ch, n)
            device=sep.device,
            shifts=sep.shifts,
            split=True,                 # 分块，否则长曲直接爆内存
            overlap=sep.overlap,
            progress=False,
        )[0]

    out = out * std + mean              # ---- 还原尺度 ----

    stems = {}
    for i, name in enumerate(sep.sources):
        y = out[i].cpu().numpy().T.astype(np.float32)   # → (n, ch)
        # 分块推理可能让输出长几百个点，这里严格对齐回输入长度
        if y.shape[0] != n:
            y = y[:n] if y.shape[0] > n else np.pad(y, ((0, n - y.shape[0]), (0, 0)))
        stems[name] = np.ascontiguousarray(y)
    return stems


def check_sum_invariant(mixture: np.ndarray, stems: dict[str, np.ndarray]) -> float:
    """返回 ``max|mixture - Σstems|``，作为一个廉价的健全性检查。

    Demucs 并不强制四轨之和等于混音（它不是掩码方法），所以这个值**不会**接近 0，
    典型量级是 0.0x。它的用处是抓离谱错误：
    如果算出来和混音峰值同量级，说明声部错位或归一化没还原。
    """
    total = sum(stems.values())
    n = min(mixture.shape[0], total.shape[0])
    return float(np.max(np.abs(mixture[:n] - total[:n])))
