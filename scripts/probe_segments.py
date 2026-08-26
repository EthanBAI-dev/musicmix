"""段落消融：把「多段带来的收益」拆成**平均次数**与**覆盖跨度**两个因子。

    python -m scripts.probe_segments

P5 的逐层探针给出一个很强的结果：全曲 4 段（120 秒）在 **13/13 层**上都优于
中间 30 秒，峰值 +0.0153 mAP —— 比换 3.35 倍大的基座（+0.0053）还多 2.9 倍。

但**为什么**涨，有两种互不相同的解释，它们对下一步的指导截然相反：

A. **覆盖跨度**：多段看到了副歌之外的部分，拿到了 30 秒窗口里没有的音乐信息。
   → 若成立，应当继续加大跨度（8 段、全曲）。
B. **平均次数**：时间平均是对「整首歌的平均特征」的一个估计量，
   多段只是让这个估计的**方差变小**，并没有新信息。
   → 若成立，收益会按 1/n 迅速饱和，加到 8 段几乎没用。

有一个现象已经偏向 B：收益在**第 0 层**（最原始的声学特征，还谈不上"音乐结构"）
就有 +0.0166，和高层一样大。如果是 A，低层不该涨这么多。

**本脚本用同一批缓存把两者分开**，不需要重提特征 ——
``*_x4`` 的逐帧特征是 4 段沿时间轴拼接的，可以切回单段任意组合：

===============  ====  =========================  ==================
组合             段数  跨度                       用来回答
===============  ====  =========================  ==================
s0 / s1 / s2 / s3   1  各自 30 秒                 单段基线（位置的影响）
s0+s1               2  前 1/3                     ┐ 段数相同、跨度不同
s1+s2               2  中段                       ├ **A 与 B 的判别对照**
s0+s3               2  **全曲两端**               ┘
s0+s1+s2+s3         4  全曲                       上界
===============  ====  =========================  ==================

**判别标准**：``s0+s3``（跨度最大）与 ``s0+s1``（跨度最小）段数都是 2。
若 A 成立，前者应明显更好；若 B 成立，两者应当打平。
"""

from __future__ import annotations

import argparse

import json
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split
from src.eval.tagging import macro_average_precision, macro_roc_auc
from src.tagging.backbone import (
    MERT_95M,
    BackboneConfig,
    frame_cache_path,
    layer_cache_path,
)

# 段组合：(名字, 用哪几段, 人类可读的跨度描述)
COMBOS: list[tuple[str, tuple[int, ...], str]] = [
    ("s0",       (0,),        "开头 30s"),
    ("s1",       (1,),        "1/3 处 30s"),
    ("s2",       (2,),        "2/3 处 30s"),
    ("s3",       (3,),        "结尾 30s"),
    ("s0+s1",    (0, 1),      "前段 · 跨度小"),
    ("s1+s2",    (1, 2),      "中段 · 跨度小"),
    ("s0+s3",    (0, 3),      "**两端 · 跨度最大**"),
    ("s0+s2",    (0, 2),      "跨度中等"),
    ("all4",     (0, 1, 2, 3), "全曲"),
]


def load_segment_means(tracks, root: Path, cfg: BackboneConfig, n_seg: int):
    """读 x4 逐帧特征，切成 n_seg 段，各段**分别**做时间平均。

    返回 (X, keep)，X 形状 (n_tracks, n_seg, dim)。
    分段平均而不是整体平均，是为了后面能任意组合 —— 组合 = 对选中的段再平均。
    """
    xs, keep = [], []
    for i, t in enumerate(tracks):
        p = frame_cache_path(t.path, root, cfg)
        if not p.exists():
            continue
        f = np.load(p).astype(np.float32)
        if f.shape[0] % n_seg:
            # 拼接长度必须能整除段数，否则说明缓存不是这个段数提的
            raise ValueError(f"{p.name}: {f.shape[0]} 帧无法整除 {n_seg} 段")
        xs.append(f.reshape(n_seg, -1, f.shape[1]).mean(axis=1))
        keep.append(i)
    if not xs:
        raise FileNotFoundError(
            "没有多段逐帧缓存。先跑：\n"
            "  python -m scripts.extract_backbone --segments 4")
    return np.stack(xs), np.array(keep)


def load_center_mean(tracks, root: Path, cfg: BackboneConfig, layer: int):
    """读**单段（中间 30 秒）**的逐层平均缓存，取指定层。

    这是项目一路用下来的默认窗口，必须放进同一个配对框架里比 ——
    否则"多段更好"这个说法的对照组是隐含的、没有误差棒的。
    """
    one = BackboneConfig(name=cfg.name, layer=layer, clip_seconds=cfg.clip_seconds,
                         n_segments=1)
    xs, keep = [], []
    for i, t in enumerate(tracks):
        q = layer_cache_path(t.path, root, one)
        if q.exists():
            xs.append(np.load(q).astype(np.float32)[layer])
            keep.append(i)
    return (np.stack(xs), np.array(keep)) if xs else (None, None)


def probe(Xtr, ytr, Xva, yva, max_iter: int):
    """返回 (mAP, ROC-AUC, 验证集分数矩阵, 有效标签掩码)。

    分数矩阵要留下来 —— 单次探针只给一个点估计，
    而本脚本里要比的差值（~0.01）和验证集抽样噪声同量级，
    必须对**歌**做自助法才能判断差值是否可区分。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.preprocessing import StandardScaler

    # 标准化只用训练集拟合 —— 用全体数据就是信息泄漏
    sc = StandardScaler().fit(Xtr)
    Xtr, Xva = sc.transform(Xtr), sc.transform(Xva)
    valid = (ytr.sum(axis=0) > 0) & (ytr.sum(axis=0) < len(ytr))
    clf = MultiOutputClassifier(LogisticRegression(max_iter=max_iter, C=1.0), n_jobs=-1)
    clf.fit(Xtr, ytr[:, valid])
    scores = np.stack([e[:, 1] for e in clf.predict_proba(Xva)], axis=1)
    return (float(macro_average_precision(yva[:, valid], scores)),
            float(macro_roc_auc(yva[:, valid], scores)), scores, valid)


def bootstrap_map_diff(y, sa, sb, n_boot: int = 1000, seed: int = 0):
    """对**歌**重采样，估计 mAP(b) − mAP(a) 的 95% CI 与方向一致率。

    两个组合用的是同一批歌、同一批标签，只有输入窗口不同 —— 是配对比较，
    所以每次重采样对两者用**同一组索引**。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        keep = (yy.sum(axis=0) > 0) & (yy.sum(axis=0) < len(yy))
        if not keep.any():
            continue
        diffs.append(macro_average_precision(yy[:, keep], sb[idx][:, keep])
                     - macro_average_precision(yy[:, keep], sa[idx][:, keep]))
    d = np.asarray(diffs)
    return float(d.mean()), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975)), \
        float((d > 0).mean())


def main() -> int:
    p = argparse.ArgumentParser(description="段落消融：平均次数 vs 覆盖跨度")
    p.add_argument("--model", default=MERT_95M)
    p.add_argument("--layer", type=int, default=6, help="x4 缓存提取时用的层")
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--subset", default="autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--max-iter", type=int, default=400)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--vs-model", default="",
                   help="再加一个基座做对照（读它的单段逐层平均），"
                        "用来把「多看歌」与「换大模型」放进同一个配对检验")
    p.add_argument("--vs-layer", type=int, default=6, help="--vs-model 用第几层")
    p.add_argument("--out", default="results/p5_segment_ablation")
    args = p.parse_args()

    root = Path(args.root)
    cfg = BackboneConfig(name=args.model, layer=args.layer, n_segments=args.segments)
    parts, vocab = load_split(args.subset, args.split, root=root, only_local=True)

    data = {}
    for name in ("train", "validation"):
        X, keep = load_segment_means(parts[name], root, cfg, args.segments)
        y = vocab.encode([parts[name][i] for i in keep])
        data[name] = (X, y)
        print(f"{name:<11}{X.shape[0]:>6} 首  {X.shape[1]} 段 × {X.shape[2]} 维")

    print(f"\n基座 {cfg.name} 第 {args.layer} 层，逐段时间平均后按组合再平均\n")

    rows, scores = [], {}
    for name, segs, span in COMBOS:
        Xtr = data["train"][0][:, list(segs)].mean(axis=1)
        Xva = data["validation"][0][:, list(segs)].mean(axis=1)
        m, a, sc, valid = probe(Xtr, data["train"][1], Xva, data["validation"][1],
                                args.max_iter)
        scores[name] = (sc, valid)
        rows.append({"combo": name, "n_segments": len(segs), "span": span,
                     "map": m, "roc_auc": a})
        print(f"  {name:<12}n={len(segs)}  {span:<22}mAP={m:.4f}  ROC-AUC={a:.4f}")

    # 把「中间 30 秒」也跑一遍，放进同一个配对框架
    ctr = {}
    for name in ("train", "validation"):
        X, keep = load_center_mean(parts[name], root, cfg, args.layer)
        ctr[name] = (X, vocab.encode([parts[name][i] for i in keep])) if X is not None else None
    if ctr["train"] and ctr["validation"]:
        if len(ctr["validation"][0]) != len(data["validation"][0]):
            print(f"  ⚠️ center 曲目数 {len(ctr['validation'][0])} 与多段 "
                  f"{len(data['validation'][0])} 不一致，跳过 center 对照（无法配对）")
        else:
            m, a, sc, valid = probe(ctr["train"][0], ctr["train"][1],
                                    ctr["validation"][0], ctr["validation"][1], args.max_iter)
            scores["center"] = (sc, valid)
            rows.insert(0, {"combo": "center", "n_segments": 1,
                            "span": "中间 30s（项目默认）", "map": m, "roc_auc": a})
            print(f"  {'center':<12}n=1  {'中间 30s（项目默认）':<22}mAP={m:.4f}  ROC-AUC={a:.4f}")
    else:
        print("  ⚠️ 无单段逐层平均缓存，跳过 center 对照")

    # 另一个基座（中间 30 秒），用来和「多看歌」直接配对比较
    if args.vs_model:
        other = BackboneConfig(name=args.vs_model, layer=args.vs_layer,
                               clip_seconds=cfg.clip_seconds, n_segments=1)
        ok = True
        vs = {}
        for name in ("train", "validation"):
            X, keep = load_center_mean(parts[name], root, other, args.vs_layer)
            if X is None or len(X) != len(data[name][0]):
                ok = False
                break
            vs[name] = (X, vocab.encode([parts[name][i] for i in keep]))
        if ok:
            m, a, sc, valid = probe(vs["train"][0], vs["train"][1],
                                    vs["validation"][0], vs["validation"][1], args.max_iter)
            scores["bigger"] = (sc, valid)
            tag = args.vs_model.split("/")[-1]
            rows.insert(1, {"combo": "bigger", "n_segments": 1,
                            "span": f"{tag} 第{args.vs_layer}层 · 中间 30s",
                            "map": m, "roc_auc": a})
            print(f"  {'bigger':<12}n=1  {tag} L{args.vs_layer} 中间30s   mAP={m:.4f}")
        else:
            print(f"  ⚠️ {args.vs_model} 的缓存缺失或曲目数不一致，跳过")

    yva = data["validation"][1]
    get = lambda n: next(r for r in rows if r["combo"] == n)          # noqa: E731
    by_n = lambda n: [r for r in rows if r["n_segments"] == n]        # noqa: E731

    singles = [r for r in by_n(1) if r["combo"] != "center"]
    best_single = max(singles, key=lambda r: r["map"])
    worst_single = min(singles, key=lambda r: r["map"])

    def cmp(a: str, b: str, question: str):
        """b 相对 a 的差值 + 自助法 CI。"""
        sa, va = scores[a]
        sb, vb = scores[b]
        assert (va == vb).all(), "两个组合的有效标签集必须相同才能配对比较"
        m, lo, hi, rate = bootstrap_map_diff(yva[:, va], sa, sb, n_boot=args.n_boot)
        sig = "✅ 可区分" if lo > 0 else ("❌ 反向可区分" if hi < 0 else "⚠️ 不可区分")
        print(f"  {question}")
        print(f"    {b} − {a} = {m:+.4f}  95%CI [{lo:+.4f}, {hi:+.4f}]"
              f"  正向率 {rate:.0%}  {sig}")
        return {"question": question, "a": a, "b": b, "delta": m,
                "ci": [lo, hi], "positive_rate": rate, "distinguishable": bool(lo > 0 or hi < 0)}

    print(f"\n{'=' * 70}")
    print(f"单段位置极差：{best_single['combo']} {best_single['map']:.4f}"
          f" vs {worst_single['combo']} {worst_single['map']:.4f}"
          f"  = {best_single['map'] - worst_single['map']:.4f}")
    print(f"{'=' * 70}\n自助法配对比较（对验证集的歌重采样 {args.n_boot} 次）：\n")

    tests = [
        cmp(worst_single["combo"], best_single["combo"],
            "① 位置效应：最好的单段 vs 最差的单段（段数都是 1）"),
        # 判别 A 的正确写法：在**段数相同**的组合里，按跨度排序看是否单调。
        # 上一版这里写成 min(pairs, key=map)，而最差的恰好就是 s0+s3 自己，
        # 于是 Δ 是拿自己减自己，恒等于 0 —— 一个不会报错、结论却完全无意义的判定。
        cmp("s1+s2", "s0+s3",
            "② 跨度效应：跨度最大 vs 跨度最小（段数都是 2）"),
        cmp(best_single["combo"], "all4",
            "③ 平均效应：4 段 vs 最好的单段"),
        cmp("s2", "s0+s2",
            "④ 一个差段能否帮到一个好段：s2 + 差的 s0"),
    ]
    if "bigger" in scores and "center" in scores:
        tests += [
            cmp("center", "bigger",
                f"⑦ 换 3.35 倍大的基座（{args.vs_model.split('/')[-1]}），窗口不变"),
        ]
    if "center" in scores:
        tests += [
            cmp("center", "all4",
                "⑤ 全曲 4 段 vs 项目默认的中间 30 秒（这是之前报的那个 +0.0153）"),
            cmp("center", best_single["combo"],
                "⑥ **只把窗口挪个位置**，成本完全不变"),
        ]
    if "bigger" in scores:
        tests.append(cmp("bigger", "all4",
                         "⑧ **正面对决**：多看歌（95M 4段）vs 换大模型（330M 中间30秒）"))

    # 跨度是否单调预测性能：段数固定为 2，按跨度从小到大排
    span_order = ["s1+s2", "s0+s2", "s0+s3"]
    vals = [get(c)["map"] for c in span_order]
    mono = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    print(f"\n  跨度单调性（段数固定=2，跨度 1/3 → 2/3 → 3/3）：")
    print("    " + "  →  ".join(f"{c} {v:.4f}" for c, v in zip(span_order, vals)))
    print(f"    单调递增？{'是' if mono else '**否**'}")

    # 组合表现能否由成分段的平均质量解释
    print(f"\n  各组合 vs 其成分单段的均值：")
    smap = {r["combo"]: r["map"] for r in singles}
    for name, segs, _ in COMBOS:
        if len(segs) < 2:
            continue
        parts_mean = float(np.mean([smap[f"s{i}"] for i in segs]))
        r = get(name)
        print(f"    {name:<8}{r['map']:.4f}  成分均值 {parts_mean:.4f}"
              f"  合并增益 {r['map'] - parts_mean:+.4f}")

    step1 = float(np.mean([r["map"] for r in by_n(2)])) - float(np.mean([r["map"] for r in singles]))
    step2 = get("all4")["map"] - float(np.mean([r["map"] for r in by_n(2)]))
    print(f"\n  边际收益  1→2 段 {step1:+.4f}   2→4 段 {step2:+.4f}"
          f"   （比值 {step2/step1:.2f}）")
    print("=" * 70)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(
        {"model": cfg.name, "layer": args.layer, "rows": rows,
         "tests": tests, "span_monotonic": mono,
         "marginal": {"1to2": step1, "2to4": step2}},
        ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 段落消融：多段的收益来自「平均次数」还是「覆盖跨度」？",
        "",
        f"- 基座：`{cfg.name}` 第 {args.layer} 层，逐帧特征切段后时间平均 + 逻辑回归",
        f"- 数据：{args.subset} · split-{args.split}，"
        f"train {data['train'][0].shape[0]} / val {data['validation'][0].shape[0]} 首",
        "- 命令：`python -m scripts.probe_segments`",
        "",
        "> 用的是**同一批 4 段缓存**切出来的子集，没有重提特征 —— ",
        "> 因此段数与跨度之外的一切（基座、层、曲目、划分）都严格相同。",
        "",
        "| 组合 | 段数 | 跨度 | mAP | ROC-AUC |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| `{r['combo']}` | {r['n_segments']} | {r['span']} "
                     f"| {r['map']:.4f} | {r['roc_auc']:.4f} |")
    lines += [
        "",
        "## 配对自助法检验",
        "",
        f"对验证集的**歌**重采样 {args.n_boot} 次；两个组合共用同一组索引（配对）。",
        "",
        "| 问题 | 比较 | Δ mAP | 95% CI | 结论 |",
        "|---|---|---|---|---|",
    ]
    for t in tests:
        verdict = ("✅ 可区分" if t["ci"][0] > 0 else
                   "❌ 反向可区分" if t["ci"][1] < 0 else "⚠️ 不可区分")
        lines.append(f"| {t['question']} | `{t['b']}` − `{t['a']}` | **{t['delta']:+.4f}** "
                     f"| [{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] | {verdict} |")
    lines += [
        "",
        f"跨度单调性（段数固定 = 2）：**{'是' if mono else '否'}**。",
        "",
        f"边际收益 1→2 段 `{step1:+.4f}`，2→4 段 `{step2:+.4f}`。",
        "",
    ]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n✅ → {out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
