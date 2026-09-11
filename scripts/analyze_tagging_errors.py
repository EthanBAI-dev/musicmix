"""标签错误分析：模型到底在哪些标签上不行，以及为什么。

    python -m scripts.analyze_tagging_errors

## 为什么不能直接按 AP 排序找「最差标签」

随机打分的 AP **约等于该标签的正例率**。一个只占 1% 的稀有标签，
瞎猜 AP 就是 0.01；一个占 30% 的常见标签，瞎猜就有 0.30。
所以按原始 AP 排序，排出来的**只是稀有度排名**，不是难度排名。

本脚本同时报两个量，并用第二个判断难度：

- **AP**：绝对水平
- **提升倍数 = AP ÷ 正例率**：比瞎猜好多少倍。这个才跨标签可比

## 混淆分析也有同样的陷阱

"标签 i 在场时模型误报 j" 的比例高，可能只是因为 j **本来就爱乱报**。
所以报的是**超额误报**：``P(误报 j | i 在场) − P(误报 j)``。
并附上真值共现率 ``P(j | i)``，用来区分"真混淆"与"语义上本就相关的标签"
（比如 electronic 与 techno 本就常一起出现，那不叫混淆）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.eval.tagging import per_tag_average_precision
from src.tagging.backbone import BackboneConfig, MERT_95M, frame_cache_path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    p = argparse.ArgumentParser(description="标签错误分析")
    p.add_argument("--ckpt", default="results/best_model.pt")
    # 这个权重的 config 里**没记录**层与段数（见 DEVLOG），只能由调用方给出，
    # 下面会用特征帧数做断言，配错就当场失败而不是静默出错
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--min-support", type=int, default=20,
                   help="混淆对至少要有多少个 (i 在场, j 不在场) 样本才统计")
    p.add_argument("--out", default="results/tagging_errors")
    args = p.parse_args()

    import torch
    from scipy.stats import spearmanr
    from torch.utils.data import DataLoader

    from src.tagging.backbone import pick_device
    from src.tagging.dataset import FeatureDataset
    from src.tagging.models import AttentionPoolHead
    from src.tagging.train import predict

    ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
    names = list(ck["tag_names"])
    thr = np.asarray(ck["thresholds"], dtype=float)

    root = Path(DEFAULT_ROOT)
    parts, vocab = load_split("autotagging_top50tags", 0, root=root, only_local=True)
    if list(vocab.tags) != names:
        raise ValueError("权重里的标签顺序与当前词表不一致 —— 预测会对错标签")

    cfg = BackboneConfig(name=MERT_95M, layer=args.layer, n_segments=args.segments)
    test = [t for t in parts["test"] if frame_cache_path(t.path, root, cfg).exists()]
    paths = [frame_cache_path(t.path, root, cfg) for t in test]
    y_test = vocab.encode(test)
    y_train = vocab.encode(parts["train"])

    # 断言特征真的是这个层/段数提的：x4 缓存每首 4×449=1796 帧、768 维
    probe = np.load(paths[0], mmap_mode="r")
    expect = 449 * args.segments
    if probe.shape != (expect, 768):
        raise ValueError(f"特征形状 {probe.shape} ≠ 预期 ({expect}, 768)："
                         f"--layer/--segments 与权重训练时不一致")

    model = AttentionPoolHead(dim=768, n_tags=len(names),
                              pooling=ck["config"].get("pooling") or "mean")
    model.load_state_dict(ck["state_dict"])
    dev = pick_device("auto")
    model.to(dev)

    # num_workers=0：上次整曲评测被 OOM 杀过，这里宁可慢一点也不 fork 多份
    loader = DataLoader(FeatureDataset(paths, y_test, crop_frames=None, train=False),
                        batch_size=32, shuffle=False, num_workers=0)
    y, s = predict(model, loader, dev)
    print(f"测试集 {len(y)} 首 × {y.shape[1]} 标签，设备 {dev}")

    # ---------------- 逐标签 ----------------
    ap = per_tag_average_precision(y, s)
    prev_test = y.mean(axis=0)
    n_train = y_train.sum(axis=0).astype(int)
    lift = ap / np.maximum(prev_test, 1e-9)
    mean_ap = float(np.nanmean(ap))
    print(f"逐标签 AP 均值 {mean_ap:.4f}（应与训练日志的测试 mAP 一致）")

    # AP 与训练集频次的相关性：若很高，"最差标签"大多只是稀有标签
    ok = ~np.isnan(ap)
    rho_ap, p_ap = spearmanr(np.log(n_train[ok] + 1), ap[ok])
    rho_lift, p_lift = spearmanr(np.log(n_train[ok] + 1), lift[ok])

    rows = []
    for j, n in enumerate(names):
        cat, tag = n.split("---", 1)
        rows.append({"tag": tag, "category": cat, "ap": float(ap[j]),
                     "prevalence": float(prev_test[j]), "lift": float(lift[j]),
                     "n_train": int(n_train[j]), "n_test": int(y[:, j].sum())})

    by_ap = sorted((r for r in rows if not np.isnan(r["ap"])), key=lambda r: r["ap"])
    by_lift = sorted((r for r in rows if not np.isnan(r["lift"])), key=lambda r: r["lift"])

    # 频次控制后的"真难"标签：在 log(频次) 上回归 AP，看残差最负的
    x = np.log(n_train[ok] + 1)
    A = np.vstack([x, np.ones_like(x)]).T
    coef, *_ = np.linalg.lstsq(A, ap[ok], rcond=None)
    resid = np.full(len(names), np.nan)
    resid[ok] = ap[ok] - A @ coef
    hard = sorted((j for j in range(len(names)) if ok[j]), key=lambda j: resid[j])[:10]

    # ---------------- 分类别 ----------------
    cats = {}
    for c, idx in vocab.groups.items():
        m = ~np.isnan(ap[idx])
        cats[c] = {"n_tags": int(len(idx)), "mean_ap": float(np.nanmean(ap[idx])),
                   "mean_lift": float(np.nanmean(lift[idx][m])),
                   "mean_prevalence": float(prev_test[idx].mean())}

    # ---------------- 混淆（超额误报） ----------------
    pred = s >= thr[None, :]
    base_fp = np.array([pred[y[:, j] == 0, j].mean() if (y[:, j] == 0).any() else np.nan
                        for j in range(len(names))])
    pairs = []
    for i in range(len(names)):
        on_i = y[:, i] == 1
        if on_i.sum() < args.min_support:
            continue
        for j in range(len(names)):
            if i == j:
                continue
            mask = on_i & (y[:, j] == 0)
            if mask.sum() < args.min_support:
                continue
            fp = pred[mask, j].mean()
            pairs.append({"present": names[i].split("---", 1)[1],
                          "false_fire": names[j].split("---", 1)[1],
                          "fp_given_present": float(fp), "base_fp": float(base_fp[j]),
                          "excess": float(fp - base_fp[j]),
                          "cooccur": float(y[on_i, j].mean()),
                          "support": int(mask.sum())})
    pairs.sort(key=lambda r: -r["excess"])

    # ---------------- 打印 ----------------
    print(f"\nAP 与 log(训练频次) 的 Spearman ρ = {rho_ap:+.3f}（p={p_ap:.1e}）")
    print(f"提升倍数 与 log(训练频次) 的 ρ = {rho_lift:+.3f}（p={p_lift:.1e}）")
    print("\n按原始 AP 最低 10：")
    for r in by_ap[:10]:
        print(f"  {r['category']:<10}{r['tag']:<18}AP {r['ap']:.3f}  正例率 {r['prevalence']:.3f}"
              f"  提升 {r['lift']:>5.1f}×  训练 {r['n_train']}")
    print("\n按提升倍数最低 10（比瞎猜好得最少）：")
    for r in by_lift[:10]:
        print(f"  {r['category']:<10}{r['tag']:<18}提升 {r['lift']:>5.1f}×  AP {r['ap']:.3f}"
              f"  训练 {r['n_train']}")
    print("\n控制频次后最难 10（AP 比同频次标签的预期低得最多）：")
    for j in hard:
        c, t = names[j].split("---", 1)
        print(f"  {c:<10}{t:<18}残差 {resid[j]:+.3f}  AP {ap[j]:.3f}  训练 {n_train[j]}")
    print("\n分类别：")
    for c, v in cats.items():
        print(f"  {c:<12}{v['n_tags']:>3} 个  AP {v['mean_ap']:.3f}  "
              f"提升 {v['mean_lift']:>5.1f}×  平均正例率 {v['mean_prevalence']:.3f}")
    print("\n超额误报最多的 12 对（i 在场时误报 j，已减去 j 的基线误报）：")
    for r in pairs[:12]:
        print(f"  {r['present']:<16}→ {r['false_fire']:<16}误报 {r['fp_given_present']:.2f}"
              f"  基线 {r['base_fp']:.2f}  超额 {r['excess']:+.2f}  真值共现 {r['cooccur']:.2f}"
              f"  (n={r['support']})")

    out = ROOT / args.out
    out.with_suffix(".json").write_text(json.dumps({
        "checkpoint": args.ckpt, "layer": args.layer, "segments": args.segments,
        "n_test": int(len(y)), "mean_ap": mean_ap,
        "spearman_ap_vs_logfreq": {"rho": float(rho_ap), "p": float(p_ap)},
        "spearman_lift_vs_logfreq": {"rho": float(rho_lift), "p": float(p_lift)},
        "tags": rows, "categories": cats,
        "hardest_after_frequency": [{"tag": names[j], "residual": float(resid[j])} for j in hard],
        "confusions": pairs[:40],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ → {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
