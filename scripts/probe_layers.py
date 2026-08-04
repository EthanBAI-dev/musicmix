"""逐层线性探针：**用数据决定该取基座的哪一层**，而不是猜。

    python -m scripts.probe_layers

MERT 有 13 层输出（1 个卷积特征 + 12 个 Transformer 层）。
浅层偏声学细节、深层偏语义，哪一层对音乐标签最好是**经验问题**。
很多项目直接用最后一层，那通常不是最优的 —— 自监督模型的最后几层
往往过度特化到预训练任务（掩码 token 预测），下游反而不如中间层。

做法：对每一层的**时间平均**特征（13×768，提取时顺手存的）训一个逻辑回归，
在验证集上比 mAP。整个过程只用 CPU、几十秒，比训 13 个完整模型便宜几个数量级。

.. note::
   这是**层选择**，不是最终性能。时间平均丢掉了所有时序信息，
   所以这里的绝对数字会低于 L1/L2。但**层与层之间的相对高低**是可信的，
   这正是我们要的。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.eval.tagging import macro_average_precision, macro_roc_auc
from src.tagging.backbone import MERT_95M, BackboneConfig, layer_cache_path


def load_layer_means(tracks, root: Path, cfg: BackboneConfig):
    """→ ``(n_tracks, n_layers, dim)``，以及成功加载的曲目下标。"""
    xs, keep = [], []
    for i, t in enumerate(tracks):
        p = layer_cache_path(t.path, root, cfg)
        if p.exists():
            xs.append(np.load(p))
            keep.append(i)
    if not xs:
        raise FileNotFoundError(
            "没有逐层平均缓存。先跑：python -m scripts.extract_backbone --layer-means-only")
    return np.stack(xs).astype(np.float32), np.array(keep)


def main() -> int:
    p = argparse.ArgumentParser(description="逐层线性探针，用数据选层")
    p.add_argument("--model", default=MERT_95M)
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--clip-seconds", type=float, default=30.0)
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--out", default="results/p4_layer_probe")
    args = p.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler

    root = Path(args.root)
    cfg = BackboneConfig(name=args.model, clip_seconds=args.clip_seconds)
    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)

    data = {}
    for name in ("train", "validation"):
        X, keep = load_layer_means(parts[name], root, cfg)
        y = vocab.encode([parts[name][i] for i in keep])
        data[name] = (X, y)
        print(f"{name:<11}{X.shape[0]:>6} 首  特征 {X.shape[1]} 层 × {X.shape[2]} 维")

    n_layers = data["train"][0].shape[1]
    print(f"\n对 {n_layers} 层各训一个逻辑回归（时间平均特征），在验证集上比 mAP\n")

    rows = []
    for layer in range(n_layers):
        Xtr, ytr = data["train"][0][:, layer], data["train"][1]
        Xva, yva = data["validation"][0][:, layer], data["validation"][1]

        # 标准化只用训练集拟合 —— 用全体数据就是信息泄漏
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)

        # 训练集里没有正样本的标签跳过（逻辑回归会直接报错）
        valid = (ytr.sum(axis=0) > 0) & (ytr.sum(axis=0) < len(ytr))
        clf = MultiOutputClassifier(
            LogisticRegression(max_iter=args.max_iter, C=1.0), n_jobs=-1)
        clf.fit(Xtr, ytr[:, valid])
        # predict_proba 每个标签返回 (n, 2)，取正类那一列
        scores = np.stack([e[:, 1] for e in clf.predict_proba(Xva)], axis=1)

        m = macro_average_precision(yva[:, valid], scores)
        a = macro_roc_auc(yva[:, valid], scores)
        rows.append({"layer": layer, "map": float(m), "roc_auc": float(a)})
        print(f"  层 {layer:>2}   mAP={m:.4f}   ROC-AUC={a:.4f}")

    best = max(rows, key=lambda r: r["map"])
    worst = min(rows, key=lambda r: r["map"])
    last = rows[-1]
    print(f"\n{'=' * 62}")
    print(f"最佳：第 {best['layer']} 层  mAP={best['map']:.4f}")
    print(f"最差：第 {worst['layer']} 层  mAP={worst['map']:.4f}")
    print(f"末层：第 {last['layer']} 层  mAP={last['map']:.4f}"
          f"（比最佳低 {(best['map']-last['map'])/best['map']*100:.1f}%）")
    print("=" * 62)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# MERT 逐层线性探针：用数据选层",
        "",
        f"- 基座：`{cfg.name}`（{n_layers} 层 × {data['train'][0].shape[2]} 维）",
        f"- 数据：{args.subset} · split-{args.split}，"
        f"train {data['train'][0].shape[0]} / val {data['validation'][0].shape[0]} 首",
        "- 特征：每层的**时间平均**（丢掉时序），逻辑回归，验证集比 mAP",
        "- 命令：`python -m scripts.probe_layers`",
        "",
        "> 这是**层选择**而非最终性能。时间平均丢掉了所有时序信息，",
        "> 绝对数字会低于 L1/L2；但层与层的**相对高低**可信，这正是我们要的。",
        "",
        "| 层 | mAP | ROC-AUC |",
        "|---|---|---|",
    ]
    for r in rows:
        mark = " ★" if r["layer"] == best["layer"] else ""
        lines.append(f"| {r['layer']}{mark} | {r['map']:.4f} | {r['roc_auc']:.4f} |")
    lines += [
        "",
        "## 结论",
        "",
        f"- **最佳是第 {best['layer']} 层**（mAP {best['map']:.4f}）",
        f"- 末层（第 {last['layer']} 层）mAP {last['map']:.4f}，"
        f"比最佳低 **{(best['map']-last['map'])/best['map']*100:.1f}%**",
        f"- 最差是第 {worst['layer']} 层（mAP {worst['map']:.4f}）",
        "",
        "自监督模型的最后几层往往过度特化到预训练任务（掩码 token 预测），",
        "下游任务反而不如中间层 —— 直接用末层是常见但次优的做法。",
        f"后续 L1/L2 一律用**第 {best['layer']} 层**。",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"model": cfg.name, "best_layer": best["layer"], "layers": rows},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ → {out.with_suffix('.md')}")
    print(f"\n下一步：python -m scripts.extract_backbone --layer {best['layer']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
