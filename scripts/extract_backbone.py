"""用冻结基座（MERT）批量提特征并缓存。L1/L2 训练之前跑一次。

    python -m scripts.extract_backbone                       # 默认 MERT-95M，第 6 层
    python -m scripts.extract_backbone --layer 9
    python -m scripts.extract_backbone --layer-means-only    # 只提逐层平均（给选层用，很快）

一次前向同时产出两样：
- 指定层的**逐帧**特征（L1/L2 训练用，约 0.7 MB/首）
- **全部 13 层的时间平均**（选层研究用，约 20 KB/首）

所以先跑一次 ``--layer-means-only`` 用 ``scripts.probe_layers`` 选出最佳层，
再跑完整提取，是最省时间的顺序。

可以随时中断重跑 —— 已缓存的会跳过。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.tagging.backbone import (
    MERT_95M,
    Backbone,
    BackboneConfig,
    frame_cache_path,
    layer_cache_path,
)


def main() -> int:
    p = argparse.ArgumentParser(description="用冻结基座批量提特征")
    p.add_argument("--model", default=MERT_95M)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--frame-stride", type=int, default=5, help="75Hz → 75/stride Hz")
    p.add_argument("--clip-seconds", type=float, default=30.0)
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--device", default="auto")
    p.add_argument("--layer-means-only", action="store_true",
                   help="只缓存逐层平均（给选层用）；不写逐帧特征")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    cfg = BackboneConfig(name=args.model, layer=args.layer,
                         frame_stride=args.frame_stride, clip_seconds=args.clip_seconds)
    root = Path(args.root)

    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)
    tracks = [t for v in parts.values() for t in v]
    if args.limit:
        tracks = tracks[: args.limit]

    bb = Backbone(cfg, device=args.device)
    print(f"基座 {cfg.name}  设备={bb.device}  {bb.n_layers} 层 × {bb.dim} 维  {bb.sr} Hz")
    print(f"取第 {cfg.layer} 层，帧步长 {cfg.frame_stride}（{bb.sr and 75 // cfg.frame_stride} Hz）")
    print(f"曲目 {len(tracks)} 首"
          + ("（只提逐层平均）" if args.layer_means_only else f" → {cfg.tag()}") + "\n")

    t0 = time.perf_counter()
    done = skipped = failed = 0
    for i, tr in enumerate(tracks, 1):
        fp = frame_cache_path(tr.path, root, cfg)
        lp = layer_cache_path(tr.path, root, cfg)
        need_frames = (not args.layer_means_only) and (not fp.exists())
        need_layers = not lp.exists()
        if not need_frames and not need_layers:
            skipped += 1
        else:
            try:
                frames, layer_means = bb.extract(bb.load_audio(root / "audio" / tr.path))
                if need_frames:
                    fp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(fp, frames)
                if need_layers:
                    lp.parent.mkdir(parents=True, exist_ok=True)
                    np.save(lp, layer_means)
                done += 1
            except Exception as e:      # 单首失败不该中断整批
                failed += 1
                if failed <= 3:
                    print(f"  ⚠️  {tr.path}: {type(e).__name__}: {e}")

        if i % 200 == 0 or i == len(tracks):
            el = time.perf_counter() - t0
            print(f"  [{i}/{len(tracks)}] 新算 {done} / 跳过 {skipped} / 失败 {failed}"
                  f"   {el/60:.1f}min，ETA {el/i*(len(tracks)-i)/60:.1f}min", flush=True)

    print(f"\n完成：新算 {done}，跳过 {skipped}，失败 {failed}，"
          f"耗时 {(time.perf_counter()-t0)/60:.1f} min")
    for label, path in (("逐帧特征", root / "features" / cfg.tag()),
                        ("逐层平均", layer_cache_path("x/y.mp3", root, cfg).parent.parent
                         / layer_cache_path("x/y.mp3", root, cfg).parent.name)):
        if path.exists():
            gb = sum(f.stat().st_size for f in path.rglob("*.npy")) / 2**30
            print(f"  {label}：{path}（{gb:.2f} GB）")
    print("\n下一步：python -m scripts.probe_layers   （用数据选层）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
