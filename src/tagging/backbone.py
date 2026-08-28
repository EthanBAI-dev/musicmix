"""自监督音乐基座模型的特征提取（P4 的 L1/L2 用）。

基座**全程冻结**，只当"耳朵"用。理由见 [02-方案调研与选型]：
从零训表征模型不现实，冻结现成基座 + 自研头部才是正确的资源分配。

.. note::
   **用哪一层不是拍脑袋的。**

   MERT 有 13 层输出（1 个卷积特征 + 12 个 Transformer 层），
   不同层编码的信息差别很大：浅层偏声学细节、深层偏语义。
   哪一层对音乐标签最好，是个**经验问题**。

   所以这里做两件事：

   1. 缓存指定层的**逐帧**特征（L1/L2 训练用）
   2. 额外缓存**全部 13 层的时间平均**（13×768，每首才 20 KB）
      —— 用它跑一次廉价的逐层线性探针，**用数据选层**而不是猜

   :func:`extract` 一次前向就同时产出这两样，不用跑两遍。

约定（和 mel 那套是**两套独立约定**，别混）：

- MERT-v1-95M 要 **24 kHz 单声道**
- 每首取中间 30 秒（与 mel 缓存对齐，保证 L0 与 L1/L2 看到的是同一段音频）
- 逐帧特征按 ``frame_stride`` 做平均池化降采样：
  原生 75 Hz 对整曲级标签是过采样，降到 15 Hz 后存储省 5 倍，
  时间分辨率 67 ms 仍然远超需要
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MERT_95M = "m-a-p/MERT-v1-95M"
MERT_330M = "m-a-p/MERT-v1-330M"


@dataclass(frozen=True)
class BackboneConfig:
    name: str = MERT_95M
    layer: int = 6                 # 默认取中间层；正式取值由 scripts.probe_layers 决定
    frame_stride: int = 5          # 75 Hz → 15 Hz
    clip_seconds: float = 30.0
    n_segments: int = 1            # >1 则在全曲上均匀取多段，特征沿时间轴拼接

    def tag(self) -> str:
        """缓存目录名。参数一变就换目录，避免读到用旧参数算的特征。"""
        short = self.name.split("/")[-1]
        seg = f"_x{self.n_segments}" if self.n_segments > 1 else ""
        return f"{short}_L{self.layer}_s{self.frame_stride}_{self.clip_seconds:.0f}s{seg}"


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class Backbone:
    """加载好的冻结基座。模型常驻，避免每首歌重新加载权重。"""

    def __init__(self, cfg: BackboneConfig = BackboneConfig(), device: str = "auto"):
        import warnings
        warnings.filterwarnings("ignore")
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        self.cfg = cfg
        self.device = pick_device(device)
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(cfg.name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(cfg.name, trust_remote_code=True)
        self.model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)          # 冻结：省显存，也杜绝手滑训到基座

        self.sr = int(self.fe.sampling_rate)
        self.n_layers = int(self.model.config.num_hidden_layers) + 1
        self.dim = int(self.model.config.hidden_size)
        if not 0 <= cfg.layer < self.n_layers:
            raise ValueError(f"layer={cfg.layer} 超出范围 [0, {self.n_layers - 1}]")

    def load_audio(self, path: Path) -> np.ndarray:
        """读中间 clip_seconds 秒，重采样到基座要求的采样率，单声道。"""
        import librosa

        dur = librosa.get_duration(path=str(path))
        offset = max(0.0, (dur - self.cfg.clip_seconds) / 2)
        y, _ = librosa.load(str(path), sr=self.sr, mono=True,
                            offset=offset, duration=self.cfg.clip_seconds)
        need = int(self.cfg.clip_seconds * self.sr)
        if len(y) < need:
            y = np.pad(y, (0, need - len(y)))     # 补零而非循环，循环会造出原曲没有的结构
        return y

    def load_segments(self, path: Path) -> list[np.ndarray]:
        """在**全曲**上均匀取 ``n_segments`` 段，每段 ``clip_seconds`` 秒。

        为什么要这个：MTG-Jamendo 曲目平均 244 秒，只取中间 30 秒等于
        **只看了 12% 的内容**就要判断风格/乐器/情绪。
        多段覆盖是验证「30 秒够不够」这个假设的最直接手段，
        而且只需重提特征、不用碰模型。

        取段方式是**等间隔覆盖全曲**（含头尾留白），不是随机采样 ——
        评测要可复现。
        """
        import librosa

        n = self.cfg.n_segments
        if n <= 1:
            return [self.load_audio(path)]

        dur = librosa.get_duration(path=str(path))
        clip = self.cfg.clip_seconds
        need = int(clip * self.sr)
        span = max(0.0, dur - clip)
        # n 段的起点等间隔铺满 [0, dur-clip]；曲子太短时全部退化到 0
        starts = [span * i / (n - 1) for i in range(n)] if span > 0 else [0.0] * n

        out = []
        for st in starts:
            y, _ = librosa.load(str(path), sr=self.sr, mono=True, offset=st, duration=clip)
            if len(y) < need:
                y = np.pad(y, (0, need - len(y)))
            out.append(y)
        return out

    @torch.no_grad()
    def extract(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """一次前向同时产出两样东西。

        Returns:
            - ``frames``：``(T', dim) float16``，指定层的逐帧特征（已按 stride 池化）
            - ``layer_means``：``(n_layers, dim) float16``，**每一层**的时间平均
              —— 给逐层探针用，每首才 20 KB
        """
        x = torch.from_numpy(y).float().unsqueeze(0).to(self.device)
        out = self.model(x, output_hidden_states=True)
        hs = out.hidden_states                       # tuple[(1, T, D)] × n_layers

        layer_means = torch.stack([h[0].mean(dim=0) for h in hs])       # (n_layers, D)

        f = hs[self.cfg.layer][0]                                        # (T, D)
        s = self.cfg.frame_stride
        if s > 1:
            t = (f.shape[0] // s) * s
            f = f[:t].reshape(-1, s, f.shape[1]).mean(dim=1)             # 平均池化降采样
        return (f.cpu().numpy().astype(np.float16),
                layer_means.cpu().numpy().astype(np.float16))


def frame_cache_path(track_path: str, root: Path, cfg: BackboneConfig) -> Path:
    return root / "features" / cfg.tag() / track_path.replace(".mp3", ".npy")


def layer_cache_path(track_path: str, root: Path, cfg: BackboneConfig) -> Path:
    """逐层平均单独存一份，和 layer / frame_stride 无关，所以目录名里不带这两项。

    但**必须带 n_segments** —— 多段的逐层平均是跨段平均后的结果，
    和单段完全不是一回事。不区分的话多段提取会静默覆盖单段的缓存，
    之后的选层探针读到的就是混杂数据（而且不会报错）。
    """
    short = cfg.name.split("/")[-1]
    seg = f"_x{cfg.n_segments}" if cfg.n_segments > 1 else ""
    return (root / "features" / f"{short}_layermeans_{cfg.clip_seconds:.0f}s{seg}"
            / track_path.replace(".mp3", ".npy"))
