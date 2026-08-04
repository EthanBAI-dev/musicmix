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

    def tag(self) -> str:
        """缓存目录名。参数一变就换目录，避免读到用旧参数算的特征。"""
        short = self.name.split("/")[-1]
        return f"{short}_L{self.layer}_s{self.frame_stride}_{self.clip_seconds:.0f}s"


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
    """逐层平均单独存一份，和 layer / frame_stride 无关，所以目录名里不带这两项。"""
    short = cfg.name.split("/")[-1]
    return (root / "features" / f"{short}_layermeans_{cfg.clip_seconds:.0f}s"
            / track_path.replace(".mp3", ".npy"))
