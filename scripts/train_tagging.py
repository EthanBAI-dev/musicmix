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
from src.tagging.dataset import MelDataset, compute_norm_stats
from src.tagging.features import MelConfig
from src.tagging.models import MelCNN, count_params
from src.tagging.train import TrainConfig, save_result, train


def main() -> int:
    p = argparse.ArgumentParser(description="训练音乐标签模型")
    p.add_argument("--level", default="L0", help="阶梯级别，用于结果命名")
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", default="bce", choices=("bce", "focal", "asl"))
    p.add_argument("--crop-frames", type=int, default=512)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    root = Path(args.root)
    cfg_mel = MelConfig()
    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)

    print(f"子集 {args.subset} · split-{args.split} · {len(vocab)} 个标签")
    for k, v in parts.items():
        print(f"  {k:<11}{len(v):>6} 首")
    print(f"  标签类别 " + "、".join(f"{k}={len(v)}" for k, v in vocab.groups.items()))

    # 归一化统计**只用训练集**算 —— 用全体数据就是信息泄漏
    mean, std = compute_norm_stats(parts["train"], root, cfg_mel)
    print(f"\nmel 归一化（仅训练集）：mean={mean:.4f} std={std:.4f}")

    def make(name: str, train_mode: bool) -> DataLoader:
        ds = MelDataset(parts[name], vocab.encode(parts[name]), root, cfg_mel,
                        crop_frames=args.crop_frames, train=train_mode, mean=mean, std=std)
        # MPS 不支持 pin_memory，开了只会刷警告
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train_mode,
                          num_workers=args.workers, pin_memory=torch.cuda.is_available(),
                          drop_last=train_mode, persistent_workers=args.workers > 0)

    loaders = {n: make(n, n == "train") for n in ("train", "validation", "test")}

    model = MelCNN(n_tags=len(vocab), n_mels=cfg_mel.n_mels)
    cfg = TrainConfig(level=args.level, epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, loss=args.loss, patience=args.patience,
                      device=args.device, seed=args.seed, crop_frames=args.crop_frames,
                      num_workers=args.workers)
    print(f"模型 MelCNN  {count_params(model):,} 参数  损失={args.loss}  "
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
    if t:
        gain = t.macro_f1_tuned - t.macro_f1_default
        print(f"\n阈值优化带来的 Macro-F1 提升：{gain:+.4f}"
              f"（mAP 不变，因为它与阈值无关）")
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
