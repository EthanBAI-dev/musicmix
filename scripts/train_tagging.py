"""训练音乐标签模型，产出 L0→L4 阶梯的一行。

    python -m scripts.train_tagging --level L0
    python -m scripts.train_tagging --level L0 --loss asl --epochs 40

每一级只改变一个因素，这样提升才能干净归因（见 :mod:`src.tagging.train`）。
结果落到 ``results/p4_<level>.json``，用 ``scripts.build_tagging_ladder``
汇总成阶梯表。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.tagging.backbone import MERT_95M, BackboneConfig, frame_cache_path
from scripts.extract_stem_features import stem_cache_path
from src.tagging.dataset import (
    FeatureDataset, MelDataset, StemFeatureDataset, compute_norm_stats)
from src.tagging.features import MelConfig
from src.tagging.models import (
    AttentionPoolHead, LinearProbe, MelCNN, StemFusionHead, count_params)
from src.tagging.train import TrainConfig, save_result, train


def main() -> int:
    p = argparse.ArgumentParser(description="训练音乐标签模型")
    p.add_argument("--level", default="L0", help="阶梯级别，用于结果命名")
    p.add_argument("--arch", default="",
                   help="melcnn(L0) / linear(L1) / attnhead(L2)。留空则按 --level 自动选")
    p.add_argument("--backbone", default=MERT_95M)
    p.add_argument("--layer", type=int, default=7,
                   help="取基座第几层。默认 7 —— 由 scripts.probe_layers 用数据选出，不是猜的")
    p.add_argument("--frame-stride", type=int, default=5)
    p.add_argument("--segments", type=int, default=1,
                   help="用几段拼接的特征（要与 extract_backbone --segments 一致）")
    p.add_argument("--fusion", default="gate", choices=("concat", "gate", "sum"),
                   help="L5 的融合方式。sum 零额外参数，是「分离是否提供新信息」的最强证据")
    p.add_argument("--pooling", default="",
                   help="mean / max / attention。留空则 L1 用 mean、L2 用 attention")
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="bce", choices=("bce", "focal", "asl"))
    p.add_argument("--crop-frames", type=int, default=512)
    p.add_argument("--use-segments", default="",
                   help="多段缓存里只用哪几段，逗号分隔（如 2 或 0,2）。空=全用。"
                        "用来在**同一批缓存**上做段落消融")
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    use_segs = (tuple(int(v) for v in args.use_segments.split(","))
                if args.use_segments.strip() else None)
    root = Path(args.root)
    cfg_mel = MelConfig()
    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)

    # 阶梯与架构的默认对应；显式传 --arch 可以覆盖（做消融时用）
    arch = args.arch or {"L0": "melcnn", "L5": "stemfusion"}.get(args.level.upper(), "attnhead")
    if not args.arch and args.level.upper() == "L1":
        arch = "linear"
    pooling = args.pooling or ("mean" if arch == "linear" else "attention")

    print(f"子集 {args.subset} · split-{args.split} · {len(vocab)} 个标签")
    for k, v in parts.items():
        print(f"  {k:<11}{len(v):>6} 首")
    print(f"  标签类别 " + "、".join(f"{k}={len(v)}" for k, v in vocab.groups.items()))

    def wrap(ds, train_mode: bool) -> DataLoader:
        # MPS 不支持 pin_memory，开了只会刷警告
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode,
                          num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                          drop_last=train_mode, persistent_workers=args.workers > 0)

    if arch == "melcnn":
        # 归一化统计**只用训练集**算 —— 用全体数据就是信息泄漏
        mean, std = compute_norm_stats(parts["train"], root, cfg_mel)
        print(f"\nmel 归一化（仅训练集）：mean={mean:.4f} std={std:.4f}")
        loaders = {
            n: wrap(MelDataset(parts[n], vocab.encode(parts[n]), root, cfg_mel,
                               crop_frames=args.crop_frames, train=(n == "train"),
                               mean=mean, std=std), n == "train")
            for n in ("train", "validation", "test")
        }
        model = MelCNN(n_tags=len(vocab), n_mels=cfg_mel.n_mels)
        desc = f"MelCNN(mel {cfg_mel.n_mels})"
    else:
        cfg_bb = BackboneConfig(name=args.backbone, layer=args.layer,
                                frame_stride=args.frame_stride, n_segments=args.segments)
        # 缺特征的曲目直接剔除，而不是训练时才 FileNotFoundError
        kept, dim = {}, None
        for n in ("train", "validation", "test"):
            paths, keep = [], []
            for i, t in enumerate(parts[n]):
                fp = frame_cache_path(t.path, root, cfg_bb)
                if fp.exists():
                    paths.append(fp); keep.append(i)
            if not paths:
                raise FileNotFoundError(
                    f"{n} 没有任何基座特征缓存（{cfg_bb.tag()}）。\n"
                    f"先跑：python -m scripts.extract_backbone --layer {args.layer}")
            dim = dim or int(np.load(paths[0]).shape[-1])
            y = vocab.encode([parts[n][i] for i in keep])
            kept[n] = wrap(FeatureDataset(paths, y, crop_frames=None,
                                          n_segments=args.segments,
                                          use_segments=use_segs,
                                          train=(n == "train")), n == "train")
            if len(paths) < len(parts[n]):
                print(f"  ⚠️  {n}: {len(parts[n]) - len(paths)} 首缺特征，已剔除")
        loaders = kept
        if arch == "stemfusion":
            # 顺序固定：mixture 在前，4 个 stem 按 StemFeatureDataset.SOURCES 排
            kept2 = {}
            for n in ("train", "validation", "test"):
                rows, keep = [], []
                for i, t in enumerate(parts[n]):
                    ps = [frame_cache_path(t.path, root, cfg_bb)] + [
                        stem_cache_path(t.path, root, cfg_bb, s_) for s_ in ("vocals", "drums", "bass", "other")]
                    if all(x.exists() for x in ps):
                        rows.append(ps); keep.append(i)
                if not rows:
                    raise FileNotFoundError(
                        f"{n} 没有完整的 stem 特征。先跑：python -m scripts.extract_stem_features")
                if len(rows) < len(parts[n]):
                    print(f"  ⚠️  {n}: {len(parts[n]) - len(rows)} 首缺 stem 特征，已剔除")
                y = vocab.encode([parts[n][i] for i in keep])
                kept2[n] = wrap(StemFeatureDataset(rows, y), n == "train")
            loaders = kept2
            model = StemFusionHead(dim=dim, n_tags=len(vocab), sources=5,
                                   fusion=args.fusion, pooling=pooling)
            desc = f"StemFusionHead(dim={dim}, fusion={args.fusion}, pooling={pooling})"
        elif arch == "linear":
            model = LinearProbe(dim=dim, n_tags=len(vocab), pooling=pooling)
            desc = f"LinearProbe(dim={dim}, pooling={pooling})"
        else:
            model = AttentionPoolHead(dim=dim, n_tags=len(vocab), pooling=pooling)
            desc = f"AttentionPoolHead(dim={dim}, pooling={pooling})"
        print(f"\n基座 {cfg_bb.name} 第 {args.layer} 层（冻结），特征目录 {cfg_bb.tag()}")
    cfg = TrainConfig(level=args.level, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, loss=args.loss, patience=args.patience,
                      device=args.device, seed=args.seed, crop_frames=args.crop_frames,
                      num_workers=args.workers)
    print(f"模型 {desc}  {count_params(model):,} 参数  损失={args.loss}  "
          f"设备={torch.cuda.is_available() and 'cuda' or (torch.backends.mps.is_available() and 'mps' or 'cpu')}\n")

    result = train(model, loaders["train"], loaders["validation"], cfg,
                   tag_groups=vocab.groups, test_loader=loaders["test"])

    out = Path(args.out or f"results/p4_{args.level.lower()}.json")
    save_result(result, out, list(vocab.tags))

    v, t = result.val_scores, result.test_scores
    print(f"\n{'=' * 74}")
    print(f"{args.level}  最佳 epoch {result.best_epoch}  训练 {result.train_seconds/60:.1f} min")
    print(f"{'=' * 74}")
    print(f"{'':12}{'ROC-AUC':>9}{'mAP':>9}{'MacroF1@0.5':>13}{'MacroF1@tuned':>15}{'MicroF1':>9}")
    for name, s in (("validation", v), ("test", t)):
        if s:
            print(f"{name:<12}{s.roc_auc:>9.4f}{s.map:>9.4f}{s.macro_f1_default:>13.4f}"
                  f"{s.macro_f1_tuned:>15.4f}{s.micro_f1_tuned:>9.4f}")
    if t and t.groups:
        print(f"\n分类别（test）：")
        for g, gs in t.groups.items():
            print(f"  {g:<12} 标签 {gs.n_valid_tags:>2}/{gs.n_tags:<2}  "
                  f"mAP={gs.map:.4f}  MacroF1@tuned={gs.macro_f1_tuned:.4f}")
    w = getattr(model, "source_weights", lambda: None)()
    if w is not None:
        print("\n各来源的门控权重（模型认为哪个更有用）：")
        for name, val in zip(StemFeatureDataset.SOURCES, w.tolist(), strict=True):
            print(f"  {name:<9}{val:.4f}")

    if t:
        gain = t.macro_f1_tuned - t.macro_f1_default
        print(f"\n阈值优化带来的 Macro-F1 提升：{gain:+.4f}"
              f"（mAP 不变，因为它与阈值无关）")
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
