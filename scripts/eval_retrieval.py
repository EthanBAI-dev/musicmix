"""相似曲目检索评测：三种嵌入 + 精确/近似索引的权衡表。

    python -m scripts.eval_retrieval

**代理真值**：两首歌共享 ≥3 个标签就算"相关"。这是弱标注，不是人工听审 ——
它衡量的是"检索结果在标签语义上是否接近"，**不等于听起来像**。
路线图里的「人工听感抽查 20 组」是另一件必须单独做的事，这里给不出来。

三种嵌入回答同一个问题：**为标签任务训练的表征，检索会更好吗？**
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.eval.retrieval import (
    build_relevance,
    evaluate_retrieval,
    ideal_gains,
    index_recall,
    shared_tag_counts,
)
from src.retrieval.index import (
    ExactIndex,
    IVFIndex,
    embed_from_features,
    embed_with_head,
)
from src.tagging.backbone import BackboneConfig, MERT_95M, frame_cache_path


def main() -> int:
    p = argparse.ArgumentParser(description="相似检索评测")
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--split-name", default="test", choices=("test", "validation"))
    p.add_argument("--ckpt", default="results/best_model.pt")
    p.add_argument("--min-shared", type=int, default=3)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--out", default="results/p7_retrieval.md")
    args = p.parse_args()

    root = Path(args.root)
    cfg = BackboneConfig(name=MERT_95M, layer=args.layer, n_segments=args.segments)
    parts, vocab = load_split("autotagging_top50tags", 0, root=root, only_local=True)

    tracks, paths = [], []
    for t in parts[args.split_name]:
        q = frame_cache_path(t.path, root, cfg)
        if q.exists():
            tracks.append(t)
            paths.append(q)
    labels = vocab.encode(tracks)
    print(f"语料 {len(tracks)} 首（{args.split_name} 划分）· 标签 {labels.shape[1]} 个")
    print(f"每首平均 {labels.sum(axis=1).mean():.1f} 个标签\n")

    # 代理真值：共享标签数矩阵（对角线是自己，后面检索会排除）
    shared = shared_tag_counts(labels, labels)
    np.fill_diagonal(shared, 0)
    n_rel = (shared >= args.min_shared).sum(axis=1)
    print(f"代理真值：共享 ≥{args.min_shared} 个标签算相关，"
          f"每首平均 {n_rel.mean():.0f} 个相关曲目"
          f"（占全库 {n_rel.mean()/len(tracks)*100:.1f}%）\n")
    if n_rel.mean() < 1:
        print("⚠️ 相关曲目过少，Recall@K 会失去意义。考虑降低 --min-shared")

    # ---------- 三种嵌入 ----------
    embs: dict[str, np.ndarray] = {}
    t0 = time.perf_counter()
    embs["mert"] = embed_from_features(paths)
    print(f"  mert    {embs['mert'].shape}  {time.perf_counter()-t0:.1f}s")
    ck = Path(args.ckpt)
    if ck.exists():
        for name in ("hidden", "logits"):
            t0 = time.perf_counter()
            embs[name] = embed_with_head(paths, ck, layer=name)
            print(f"  {name:<8}{embs[name].shape}  {time.perf_counter()-t0:.1f}s")
    else:
        print(f"  ⚠️ 没有 {ck}，跳过监督嵌入。先跑："
              f"\n     python -m scripts.train_tagging --level L2 --arch attnhead "
              f"--pooling mean --layer 6 --segments 4 --save-model --out results/best_model.json")

    KS = (1, 5, 10, 50)
    rows = []

    # ---- 随机基线：没有它，R@10=0.14 读不出是好是坏 ----
    # 这和 P1 给分离设 trivial/silence 下界是同一件事：
    # 一个指标必须先有锚点，数字才有含义。
    rng = np.random.default_rng(0)
    rand_idx = np.stack([rng.permutation(len(tracks))[: max(KS)] for _ in range(len(tracks))])
    rel_r, gain_r, nrel_r = build_relevance(rand_idx, shared, args.min_shared)
    ideal_r = ideal_gains(shared, max(KS))
    sc_r = evaluate_retrieval(rel_r, nrel_r, gain_r, ideal_r, ks=KS)
    rows.append({"embedding": "random（下界）", "dim": 0, "ms": 0.0,
                 **{f"recall@{k}": sc_r.recall[k] for k in KS},
                 "map@10": sc_r.map[10], "ndcg@10": sc_r.ndcg[10]})
    exact_idx: dict[str, np.ndarray] = {}
    print(f"\n{'嵌入':<10}{'维度':>6}{'R@1':>8}{'R@10':>8}{'R@50':>8}"
          f"{'mAP@10':>9}{'NDCG@10':>9}{'耗时ms':>8}")
    print(f"  {'random':<8}{'—':>6}{sc_r.recall[1]:>8.4f}{sc_r.recall[10]:>8.4f}"
          f"{sc_r.recall[50]:>8.4f}{sc_r.map[10]:>9.4f}{sc_r.ndcg[10]:>9.4f}{'—':>8}")
    for name, E in embs.items():
        idx, st = ExactIndex(E).search(E, k=max(KS))
        exact_idx[name] = idx
        rel, gain, nrel = build_relevance(idx, shared, args.min_shared)
        ideal = ideal_gains(shared, max(KS))
        sc = evaluate_retrieval(rel, nrel, gain, ideal, ks=KS)
        r = {"embedding": name, "dim": int(E.shape[1]), "ms": st.seconds * 1000,
             **{f"recall@{k}": sc.recall[k] for k in KS},
             "map@10": sc.map[10], "ndcg@10": sc.ndcg[10]}
        rows.append(r)
        print(f"  {name:<8}{E.shape[1]:>6}{sc.recall[1]:>8.4f}{sc.recall[10]:>8.4f}"
              f"{sc.recall[50]:>8.4f}{sc.map[10]:>9.4f}{sc.ndcg[10]:>9.4f}"
              f"{st.seconds*1000:>8.1f}")

    # ---------- 精确 vs 近似 ----------
    best = max((r for r in rows if r["dim"] > 0),
               key=lambda r: r["recall@10"])["embedding"]
    print(f"\n用最好的嵌入（{best}）做精确/近似权衡：")
    print(f"{'方法':<18}{'比较次数':>12}{'耗时ms':>9}{'索引召回@10':>13}")
    E = embs[best]
    _, st_ex = ExactIndex(E).search(E, k=10)
    print(f"  {'精确（矩阵乘）':<16}{len(E)**2:>12,}{st_ex.seconds*1000:>9.1f}{1.0:>13.3f}")
    trade = [{"method": "exact", "n_compared": len(E) ** 2,
              "ms": st_ex.seconds * 1000, "index_recall@10": 1.0}]
    ivf = IVFIndex(E, n_lists=64)
    for probe in (1, 4, 16):
        idx_a, st_a = ivf.search(E, k=10, n_probe=probe)
        rec = index_recall(idx_a, exact_idx[best][:, :10], 10)
        print(f"  {'IVF n_probe=' + str(probe):<16}{st_a.n_compared:>12,}"
              f"{st_a.seconds*1000:>9.1f}{rec:>13.3f}")
        trade.append({"method": f"ivf_probe{probe}", "n_compared": int(st_a.n_compared),
                      "ms": st_a.seconds * 1000, "index_recall@10": float(rec)})

    # ---------- 写文件 ----------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"n_tracks": len(tracks), "split": args.split_name,
         "min_shared": args.min_shared, "embeddings": rows, "tradeoff": trade},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 相似曲目检索",
        "",
        f"- 语料：{len(tracks)} 首（`{args.split_name}` 划分）",
        f"- **代理真值**：共享 ≥{args.min_shared} 个标签算相关，"
        f"每首平均 {n_rel.mean():.0f} 个相关曲目",
        "",
        "> 代理真值是**弱标注**，衡量的是「检索结果在标签语义上是否接近」，",
        "> **不等于听起来像**。人工听感抽查是另一件必须单独做的事，这里给不出来。",
        "",
        "## 三种嵌入",
        "",
        "| 嵌入 | 维度 | 监督 | R@1 | R@10 | R@50 | mAP@10 | NDCG@10 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    sup = {"mert": "无（冻结基座）", "hidden": "弱（50 标签）", "logits": "弱（50 标签）",
           "random（下界）": "—"}
    for r in rows:
        lines.append(
            f"| `{r['embedding']}` | {r['dim']} | {sup.get(r['embedding'], '')} "
            f"| {r['recall@1']:.4f} | {r['recall@10']:.4f} | {r['recall@50']:.4f} "
            f"| {r['map@10']:.4f} | {r['ndcg@10']:.4f} |")
    lines += ["", "## 精确 vs 近似（IVF）", "",
              f"用 `{best}` 嵌入。**索引召回**= 近似结果与精确结果的重合率。", "",
              "| 方法 | 比较次数 | 耗时 (ms) | 索引召回@10 |", "|---|---|---|---|"]
    for t in trade:
        lines.append(f"| {t['method']} | {t['n_compared']:,} | {t['ms']:.1f} "
                     f"| {t['index_recall@10']:.3f} |")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
