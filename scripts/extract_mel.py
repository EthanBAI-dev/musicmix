"""批量提取 mel 频谱并缓存。L0 训练之前必须先跑一次。

    python -m scripts.extract_mel --subset autotagging_top50tags --workers 8

为什么单独一步而不是训练时现算：mp3 解码 + mel 变换约 0.3 s/首（CPU 单核），
5500 首就是半小时。训练要跑几十个 epoch，每轮重算完全不可接受。
缓存成 float16 的 ``.npy``，每首约 470 KB。

可以随时中断重跑 —— 已缓存的会跳过。
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.tagging.features import MelConfig, cache_path, compute_mel


def _one(args: tuple[str, str, MelConfig]) -> tuple[str, bool, str]:
    track_path, root, cfg = args
    root = Path(root)
    dst = cache_path(track_path, root, cfg)
    if dst.exists():
        return track_path, True, "cached"
    try:
        import numpy as np
        mel = compute_mel(root / "audio" / track_path, cfg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(dst, mel)
        return track_path, True, "ok"
    except Exception as e:                       # 单首失败不该中断整批
        return track_path, False, f"{type(e).__name__}: {e}"


def main() -> int:
    p = argparse.ArgumentParser(description="批量提取并缓存 mel 特征")
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--n-mels", type=int, default=128)
    p.add_argument("--clip-seconds", type=float, default=30.0)
    args = p.parse_args()

    cfg = MelConfig(n_mels=args.n_mels, clip_seconds=args.clip_seconds)
    root = Path(args.root)

    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)
    tracks = [t for v in parts.values() for t in v]
    print(f"子集 {args.subset} · split-{args.split}")
    print(f"本地曲目 {len(tracks)}（train {len(parts['train'])} / "
          f"val {len(parts['validation'])} / test {len(parts['test'])}），标签 {len(vocab)} 个")
    print(f"特征配置 {cfg.tag()} → {cfg.n_mels}×{cfg.n_frames}\n")

    todo = [(t.path, str(root), cfg) for t in tracks]
    t0 = time.perf_counter()
    ok = cached = 0
    failures: list[tuple[str, str]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_one, a) for a in todo]
        for i, f in enumerate(as_completed(futs), 1):
            path, success, msg = f.result()
            if not success:
                failures.append((path, msg))
            elif msg == "cached":
                cached += 1
            else:
                ok += 1
            if i % 500 == 0 or i == len(futs):
                el = time.perf_counter() - t0
                print(f"  [{i}/{len(futs)}] 新算 {ok} / 已缓存 {cached} / 失败 {len(failures)}"
                      f"   {el:.0f}s，ETA {el/i*(len(futs)-i):.0f}s", flush=True)

    out = root / "features" / cfg.tag()
    size = sum(f.stat().st_size for f in out.rglob("*.npy")) / 2**30 if out.exists() else 0
    print(f"\n完成：新算 {ok}，已缓存 {cached}，失败 {len(failures)}")
    print(f"缓存目录 {out}（{size:.2f} GB）")
    if failures:
        print(f"\n⚠️  {len(failures)} 首失败，前 5 条：")
        for path, msg in failures[:5]:
            print(f"   {path}: {msg}")
        print("   （这些曲目会在训练时被跳过；多半是 mp3 损坏）")
    print("\n下一步：python -m scripts.train_tagging --level L0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
