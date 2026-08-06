"""Stem-aware 特征提取：分离 → 逐 stem 提基座特征（M5 / L5 的前置步骤）。

    python -m scripts.extract_stem_features --limit 200      # 先小规模验证
    python -m scripts.extract_stem_features                  # 全量

这是本项目**唯一的原创点**（[00-总体大纲] P4-L5）的数据准备：
把每首歌先用 htdemucs 分成 4 个 stem，再对每个 stem 单独提 MERT 特征。
L5 把这 4 份特征和原混音特征融合，检验假设：

- **H1**：Stem-aware 融合能提升标签性能
- **H2**（更强、更有意思）：**提升主要来自乐器类标签**，风格次之，情绪最少

为什么 H2 值得单独提出来：如果只报总体 mAP 涨了，无法区分"真的因为分离而受益"
还是"多了 4 倍特征量所以涨了"。**按标签类别拆分的收益归因才是证据**。

.. note::
   **只处理 30 秒片段，且与 mel / 混音特征取同一段。**
   全曲分离要 214 小时（5,404 首 × 244 秒平均 × RTF 0.066），完全不现实；
   30 秒片段约 3 小时。窗口对齐由 :func:`center_window` 保证 ——
   如果 L5 用的音频段和 L1/L2 不一致，比较就没有意义了。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.tagging.backbone import MERT_95M, Backbone, BackboneConfig

STEMS = ("vocals", "drums", "bass", "other")


def stem_cache_path(track_path: str, root: Path, cfg: BackboneConfig, stem: str) -> Path:
    """``"14/214.mp3"`` → ``<root>/features/<tag>_stem-<stem>/14/214.npy``"""
    return (root / "features" / f"{cfg.tag()}_stem-{stem}"
            / track_path.replace(".mp3", ".npy"))


def center_window(y: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """取中间 seconds 秒；不足则补零。

    **必须和 features.py / backbone.py 用同一个取窗规则** ——
    L5 与 L1/L2 比较的前提是两者看的是同一段音频。
    """
    need = int(seconds * sr)
    if len(y) <= need:
        return np.pad(y, (0, need - len(y)))
    off = (len(y) - need) // 2
    return y[off : off + need]


def main() -> int:
    p = argparse.ArgumentParser(description="分离 + 逐 stem 提基座特征")
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--backbone", default=MERT_95M)
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--clip-seconds", type=float, default=30.0)
    p.add_argument("--sep-model", default="htdemucs")
    p.add_argument("--device", default="auto")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    import librosa

    from src.separation import demucs_model

    root = Path(args.root)
    cfg = BackboneConfig(name=args.backbone, layer=args.layer,
                         frame_stride=args.frame_stride, clip_seconds=args.clip_seconds)

    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)
    tracks = [t for v in parts.values() for t in v]
    if args.limit:
        tracks = tracks[: args.limit]

    sep = demucs_model.load(args.sep_model, device=args.device)
    bb = Backbone(cfg, device=args.device)
    print(f"分离 {args.sep_model}（{sep.device}）→ 基座 {cfg.name} 第 {args.layer} 层")
    print(f"曲目 {len(tracks)} 首，每首中间 {args.clip_seconds:.0f} 秒 × 4 stems\n")

    t0 = time.perf_counter()
    done = skipped = failed = 0
    for i, tr in enumerate(tracks, 1):
        outs = {s: stem_cache_path(tr.path, root, cfg, s) for s in STEMS}
        if all(p_.exists() for p_ in outs.values()):
            skipped += 1
        else:
            try:
                # 分离要 44.1kHz 立体声（demucs 的约定）
                y, _ = librosa.load(str(root / "audio" / tr.path), sr=44100, mono=False)
                if y.ndim == 1:
                    y = np.stack([y, y])
                mix = center_window(y.T, 44100, args.clip_seconds)   # → (n, 2)
                stems = demucs_model.separate(sep, mix)

                for s, dst in outs.items():
                    # 基座要 24kHz 单声道，和主流程一致
                    mono = librosa.resample(stems[s].mean(axis=1), orig_sr=44100, target_sr=bb.sr)
                    frames, _ = bb.extract(mono.astype(np.float32))
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    np.save(dst, frames)
                done += 1
            except Exception as e:
                failed += 1
                if failed <= 3:
                    print(f"  ⚠️  {tr.path}: {type(e).__name__}: {e}")

        if i % 50 == 0 or i == len(tracks):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(tracks)}] 新算 {done} / 跳过 {skipped} / 失败 {failed}"
                  f"   {el/60:.1f}min，ETA {el/i*(len(tracks)-i)/60:.0f}min", flush=True)

    gb = sum(f.stat().st_size for s in STEMS
             for f in (root / "features" / f"{cfg.tag()}_stem-{s}").rglob("*.npy")) / 2**30
    print(f"\n完成：新算 {done}，跳过 {skipped}，失败 {failed}，"
          f"耗时 {(time.perf_counter()-t0)/60:.0f} min，缓存 {gb:.1f} GB")
    print("\n下一步：python -m scripts.train_tagging --level L5 --arch stemfusion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
