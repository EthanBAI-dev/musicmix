"""蒸馏用的小型分离模型（学生）。

教师是 htdemucs（42 M 参数）。学生是一个**频谱域 U-Net**，
输出 4 个声部的复数掩码，参数量可调（默认约 6 M，小 7 倍）。

## 为什么用复数掩码而不是幅度掩码

幅度掩码要复用混音的相位，而 P2 实测过：相位误差占了掩码类方法损失的 **102%**，
IRM 理想幅度掩码的上界只有 8.95 dB。复数掩码让模型能同时修正幅度和相位，
天花板高得多。代价是更难训 —— 所以掩码不做任何有界化（不套 tanh/sigmoid），
让它自由取值。

## 为什么蒸馏能用无标注音乐

正常训练分离模型需要**分轨真值**，而这种数据极稀缺（MUSDB18-HQ 训练集只有 100 首）。
蒸馏不需要：教师跑一遍就把目标造出来了，所以**任何音乐都能当训练数据**。
本项目手上有 5,534 首 Jamendo 音频 —— 比 MUSDB18-HQ 的训练集多 55 倍。

**这才是蒸馏在这个项目里的真正意义**，不是"压缩模型"。
"""

from __future__ import annotations

import torch
import torch.nn as nn

SOURCES = ("drums", "bass", "other", "vocals")   # 顺序与 htdemucs 一致，不要改


class ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU ×2。

    用 GroupNorm 而不是 BatchNorm：训练时 batch 很小（音频张量占显存），
    BatchNorm 在小 batch 下统计量不稳，会让验证指标忽高忽低。
    """

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class StudentUNet(nn.Module):
    """频谱域 U-Net，输出 ``len(SOURCES)`` 个复数掩码。

    Args:
        n_fft: STFT 点数。1024 @ 44.1kHz ≈ 23 ms 窗，兼顾时间与频率分辨率
        hop: 帧移
        base: 第一层通道数。**这是控制模型大小的旋钮** ——
            base=32 约 6 M 参数，base=16 约 1.5 M
        depth: 下采样层数
    """

    def __init__(self, n_fft: int = 1024, hop: int = 256,
                 base: int = 32, depth: int = 4, n_sources: int = len(SOURCES)):
        super().__init__()
        self.n_fft, self.hop, self.n_sources = n_fft, hop, n_sources
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)

        chans = [base * 2 ** i for i in range(depth)]
        self.chans = chans
        # 下采样次数 = 上采样次数 = depth-1。
        # 第一版让 mid 也 pool 了一次，于是 4 次下采样对 3 次上采样，
        # 跳连的空间尺寸直接对不上（报 "expected 64 but got 130"）。
        # 瓶颈层**不改变分辨率**，只加深通道。
        self.inc = ConvBlock(4, chans[0])
        self.downs = nn.ModuleList(
            [ConvBlock(chans[i], chans[i + 1]) for i in range(depth - 1)])
        self.pool = nn.MaxPool2d(2)
        self.mid = ConvBlock(chans[-1], chans[-1])
        self.ups = nn.ModuleList(
            [nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2)
             for i in range(depth - 1)][::-1])
        self.up_blocks = nn.ModuleList(
            [ConvBlock(chans[i] * 2, chans[i]) for i in range(depth - 1)][::-1])
        # 每个声部 2 声道 × (实部, 虚部) = 4 个输出平面
        self.out = nn.Conv2d(chans[0], n_sources * 4, 1)

    # ---------- STFT ----------

    def stft(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, 2, T)`` → ``(B, 4, F, N)``（实部/虚部 × 左右声道）。"""
        b, c, t = wav.shape
        z = torch.stft(wav.reshape(b * c, t), self.n_fft, self.hop,
                       window=self.window, return_complex=True, center=True)
        z = torch.view_as_real(z)                      # (B*C, F, N, 2)
        f, n = z.shape[1], z.shape[2]
        return z.reshape(b, c, f, n, 2).permute(0, 1, 4, 2, 3).reshape(b, c * 2, f, n)

    def istft(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        """``(B, S, 4, F, N)`` → ``(B, S, 2, T)``。"""
        b, s, _, f, n = spec.shape
        z = spec.reshape(b * s, 2, 2, f, n).permute(0, 1, 3, 4, 2)   # (B*S, C, F, N, 2)
        z = torch.view_as_complex(z.reshape(b * s * 2, f, n, 2).contiguous())
        wav = torch.istft(z, self.n_fft, self.hop, window=self.window,
                          length=length, center=True)
        return wav.reshape(b, s, 2, length)

    # ---------- 前向 ----------

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """``(B, 2, T)`` 混音 → ``(B, S, 2, T)`` 四个声部。"""
        length = wav.shape[-1]
        x = self.stft(wav)                             # (B, 4, F, N)
        f0, n0 = x.shape[-2], x.shape[-1]
        # 尺寸补齐到 2^(下采样次数) 的倍数，否则跳连对不上
        m = 2 ** len(self.downs)
        pf, pn = (-f0) % m, (-n0) % m
        if pf or pn:
            x = nn.functional.pad(x, (0, pn, 0, pf))

        skips = []
        h = self.inc(x)
        for d in self.downs:
            skips.append(h)
            h = d(self.pool(h))
        h = self.mid(h)                     # 瓶颈不改分辨率

        for up, blk, skip in zip(self.ups, self.up_blocks, reversed(skips), strict=True):
            h = up(h)
            h = blk(torch.cat([h, skip], dim=1))

        mask = self.out(h)[..., :f0, :n0]               # (B, S*4, F, N)
        b = mask.shape[0]
        mask = mask.reshape(b, self.n_sources, 4, f0, n0)

        # 复数掩码乘到混音谱上。**不做有界化** —— 见模块 docstring
        mix = x[..., :f0, :n0].reshape(b, 1, 2, 2, f0, n0)     # (B,1,C,RI,F,N)
        mk = mask.reshape(b, self.n_sources, 2, 2, f0, n0)
        # 复数乘法：(a+bi)(c+di) = (ac−bd) + (ad+bc)i
        a, bb = mix[:, :, :, 0], mix[:, :, :, 1]
        c, d = mk[:, :, :, 0], mk[:, :, :, 1]
        est = torch.stack([a * c - bb * d, a * d + bb * c], dim=3)
        est = est.reshape(b, self.n_sources, 4, f0, n0)
        return self.istft(est, length)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def distill_loss(student_out: torch.Tensor, teacher_out: torch.Tensor,
                 alpha: float = 0.5) -> torch.Tensor:
    """学生对齐教师：波形 L1 + 多分辨率频谱 L1。

    只用波形 L1 会让学生学出"听起来闷"的结果 —— 波形误差对相位极敏感，
    模型会为了压低它而牺牲高频细节。加一项频谱损失能拉回来。

    Args:
        alpha: 频谱项权重。0 = 纯波形
    """
    loss = torch.nn.functional.l1_loss(student_out, teacher_out)
    if alpha <= 0:
        return loss

    b, s, c, t = student_out.shape
    a = student_out.reshape(b * s * c, t)
    e = teacher_out.reshape(b * s * c, t)
    spec = 0.0
    for n_fft in (512, 1024, 2048):
        w = torch.hann_window(n_fft, device=a.device)
        sa = torch.stft(a, n_fft, n_fft // 4, window=w, return_complex=True).abs()
        se = torch.stft(e, n_fft, n_fft // 4, window=w, return_complex=True).abs()
        spec = spec + torch.nn.functional.l1_loss(sa, se)
    return loss + alpha * spec / 3


@torch.no_grad()
def separate_chunked(model: "StudentUNet", wav: torch.Tensor, sr: int = 44100,
                     segment: float = 10.0, overlap: float = 0.25) -> torch.Tensor:
    """分段推理 + 重叠相加。整首一次前向会爆内存。

    ``(2, T)`` → ``(S, 2, T)``。

    为什么必须有这个：一首 4 分钟的歌经 STFT 后是 513 × 41,400 帧，
    U-Net 第一层 32 通道就要 **2.7 GB** 一个张量 —— 而这只是第一层。
    实测整首前向直接被系统 OOM kill（exit 137）。

    demucs 的 ``apply_model(split=True)`` 做的是同一件事；
    学生模型漏掉它，只有在**全曲**上才会暴露 —— 单测里的 5 秒片段永远碰不到。

    重叠部分用**线性淡入淡出**加权，而不是直接取平均：
    直接平均会在边界留下能量凹陷（两段各贡献一半但相位未必对齐）。
    """
    device = next(model.parameters()).device
    n = wav.shape[-1]
    seg = int(segment * sr)
    hop = max(1, int(seg * (1 - overlap)))

    if n <= seg:
        return model(wav[None].to(device))[0].cpu()

    out = torch.zeros(len(SOURCES), wav.shape[0], n)
    norm = torch.zeros(n)
    # 淡入淡出窗：中间为 1，两端线性降到 0
    ramp = int(seg * overlap / 2) or 1
    win = torch.ones(seg)
    win[:ramp] = torch.linspace(0, 1, ramp)
    win[-ramp:] = torch.linspace(1, 0, ramp)

    for start in range(0, n, hop):
        end = min(start + seg, n)
        chunk = wav[:, start:end]
        if chunk.shape[-1] < 2:
            break
        est = model(chunk[None].to(device))[0].cpu()
        w = win[: chunk.shape[-1]]
        out[..., start:end] += est * w
        norm[start:end] += w
        if end >= n:
            break

    return out / norm.clamp(min=1e-8)
