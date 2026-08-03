"""MTG-Jamendo 标签分布统计 —— 训练之前必须先看的一张图。

    python -m scripts.jamendo_stats                    # 本地已下载的子样本
    python -m scripts.jamendo_stats --all              # 全量元数据（不需要音频）
    python -m scripts.jamendo_stats --subset autotagging_top50tags

产出 ``results/p4_tag_distribution.{md,png,json}``。

为什么这一步不能跳：多标签任务的**全部难点都在标签分布里**。
不先看清楚长尾有多长，就会犯两个错：

1. 用默认阈值 0.5 —— 稀有标签的预测概率可能永远到不了 0.5，等于直接放弃它们，
   而 Macro-F1 对此极其敏感
2. 在正样本个位数的标签上做逐标签阈值搜索 —— 那是在过拟合噪声，
   验证集数字会虚高，测试集上原形毕露

所以这张图直接决定了 ``min_positives`` 该取多少。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.datasets.jamendo import DEFAULT_ROOT, load_split, load_subset, tag_statistics


def _use_cjk_font() -> None:
    """让图里的中文正常显示。

    不设的话 matplotlib 会用 DejaVu Sans，中文字符全部渲染成方框（且只在
    stderr 里刷一堆 findfont 警告，图本身"看起来正常"地生成出来）。
    """
    import matplotlib
    from matplotlib import font_manager

    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("PingFang SC", "Heiti SC", "Hiragino Sans GB",
                 "Arial Unicode MS", "Noto Sans CJK SC", "Microsoft YaHei"):
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return
    print("⚠️  没找到中文字体，图中的中文会显示为方框")


def plot(stats: dict, counts: dict[str, int], out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    _use_cjk_font()
    import matplotlib.pyplot as plt

    freq = np.array(sorted(counts.values())[::-1])
    cats = {}
    for tag, c in counts.items():
        cats.setdefault(tag.split("---")[0], []).append(c)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    fig.suptitle(title, fontsize=12)

    # ① 长尾曲线（对数纵轴）
    ax[0].plot(np.arange(1, len(freq) + 1), freq, lw=1.8, color="#2a9d8f")
    ax[0].axhline(50, color="#e76f51", ls="--", lw=1,
                  label=f"50 正样本（低于此有 {int((freq < 50).sum())} 个标签）")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("标签排名"); ax[0].set_ylabel("正样本数（对数）")
    ax[0].set_title("长尾分布"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    # ② 累积占比 —— 一眼看出"头部多少标签吃掉了多少正样本"
    cum = np.cumsum(freq) / freq.sum()
    x = np.arange(1, len(freq) + 1) / len(freq) * 100
    ax[1].plot(x, cum * 100, lw=1.8, color="#4cc9f0")
    ax[1].plot([0, 100], [0, 100], ls=":", color="gray", lw=1, label="完全均匀")
    ax[1].set_xlabel("标签占比 (%)"); ax[1].set_ylabel("累计正样本占比 (%)")
    ax[1].set_title(f"集中度：头部 10% 标签占 {stats['head10pct_share']*100:.0f}% 正样本")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    # ③ 分类别箱线
    order = sorted(cats)
    ax[2].boxplot([cats[c] for c in order], tick_labels=order, showfliers=False)
    ax[2].set_yscale("log"); ax[2].set_ylabel("正样本数（对数）")
    ax[2].set_title("各类别标签的样本量"); ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description="MTG-Jamendo 标签分布统计")
    p.add_argument("--subset", default="autotagging",
                   help="split 用的子集名：autotagging / autotagging_instrument / autotagging_top50tags")
    p.add_argument("--split", type=int, default=0)
    p.add_argument("--all", action="store_true", help="用全量元数据，忽略本地是否有音频")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--min-positives", type=int, default=0)
    p.add_argument("--out", default="results/p4_tag_distribution")
    args = p.parse_args()

    root = Path(args.root)
    if args.all:
        tracks = load_subset("autotagging_real", root=root, only_local=False)
        parts, vocab = {"（全量，未分 split）": tracks}, None
        scope = "全量元数据（55,609 首，不依赖本地音频）"
    else:
        parts, vocab = load_split(args.subset, args.split, root=root,
                                  only_local=True, min_positives=args.min_positives)
        n = sum(len(v) for v in parts.values())
        scope = f"本地子样本 · {args.subset} · split-{args.split}（{n} 首）"

    print(f"范围：{scope}\n")

    all_tracks = [t for v in parts.values() for t in v]
    overall = tag_statistics(all_tracks, vocab)

    lines = [
        "# MTG-Jamendo 标签分布",
        "",
        f"- 范围：{scope}",
        f"- 命令：`python -m scripts.jamendo_stats "
        + ("--all" if args.all else f"--subset {args.subset} --min-positives {args.min_positives}") + "`",
        "",
        "## 总览",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 曲目数 | {overall['n_tracks']:,} |",
        f"| 总时长 | {overall['total_hours']:.0f} 小时 |",
        f"| 标签数 | {overall['n_tags']} |",
        f"| 每首平均标签数 | {overall['tags_per_track']:.2f} |",
        f"| 标签矩阵密度 | {overall['label_density']*100:.2f}%（**其余 {100-overall['label_density']*100:.1f}% 是 0**）|",
        f"| 最高频标签正样本 | {overall['freq_max']:,} |",
        f"| 中位标签正样本 | {overall['freq_median']:,} |",
        f"| 最低频标签正样本 | {overall['freq_min']:,} |",
        f"| 头部 10% 标签占正样本比例 | **{overall['head10pct_share']*100:.0f}%** |",
        f"| 正样本 < 50 的标签数 | **{overall['n_below_50']} / {overall['n_tags']}** |",
        "",
        "## 各类别",
        "",
        "| 类别 | 标签数 | 最少 | 中位 | 最多 |",
        "|---|---|---|---|---|",
    ]
    for c, d in overall["by_category"].items():
        lines.append(f"| {c} | {d['n_tags']} | {d['min']:,} | {d['median']:,} | {d['max']:,} |")

    if not args.all:
        lines += ["", "## 各 split", "", "| split | 曲目数 | 时长(h) |", "|---|---|---|"]
        for k, v in parts.items():
            s = tag_statistics(v, vocab)
            lines.append(f"| {k} | {s['n_tracks']:,} | {s['total_hours']:.0f} |")

        # 阈值调优可行性：验证集正样本太少的标签，搜出来的阈值是噪声
        val = parts.get("validation", [])
        if val:
            vs = tag_statistics(val, vocab)["counts"]
            thin = sorted((n, t) for t, n in vs.items() if n < 10)
            lines += [
                "", "## 逐标签阈值调优的可行性（L4）", "",
                f"验证集共 {len(val):,} 首。**正样本 < 10 的标签有 {len(thin)} / {overall['n_tags']} 个** —— "
                "在这些标签上搜阈值等于过拟合噪声。",
                "",
                f"最稀薄的 10 个：{'、'.join(f'`{t}`({n})' for n, t in thin[:10]) or '无'}",
            ]

    top = list(overall["counts"].items())
    lines += ["", "## 最高频 15 个标签", "", "| 标签 | 正样本 | 占比 |", "|---|---|---|"]
    for t, n in top[:15]:
        lines.append(f"| `{t}` | {n:,} | {n/overall['n_tracks']*100:.1f}% |")
    lines += ["", "## 最低频 15 个标签", "", "| 标签 | 正样本 |", "|---|---|"]
    for t, n in top[-15:]:
        lines.append(f"| `{t}` | {n:,} |")

    out = Path(args.out)
    png = out.with_suffix(".png")
    plot(overall, overall["counts"], png, f"MTG-Jamendo 标签分布 · {scope}")
    lines += ["", f"![标签分布]({png.name})"]

    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({k: v for k, v in overall.items() if k != "counts"} | {"scope": scope},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(lines[5:34]))
    print(f"\n✅ → {out.with_suffix('.md')} / {png.name} / {out.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
