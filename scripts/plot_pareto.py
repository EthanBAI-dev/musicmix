"""画 SDR–RTF 帕累托曲线：分离质量与推理速度的取舍。

    python -m scripts.plot_pareto

**数据全部来自已经跑过的实验**，不需要重跑。回答一个具体问题：
*在推理期能做的几种选择里，哪些落在帕累托前沿上？*

蒸馏出来的学生**在图上**（2.17 dB / RTF 0.011）。它落在帕累托前沿上 ——
**不是因为它好，而是因为没有别的点比它更快**。
前沿上的点不等于好点，这正是帕累托图容易被误读的地方：
它只说"在这个速度下没有更准的"，不说"这个准度可用"。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "results"

# (结果文件, 展示名, 是否为可选的推理期设置)
POINTS = [
    ("p1_trivial.json", "trivial（混音当每轨）", False),
    ("p1_htdemucs.json", "htdemucs 基线", True),
    ("p2_a1_overlap50.json", "＋overlap 0.50", True),
    ("p2_a2_tta.json", "＋TTA", True),
    ("p2_a3_mask.json", "＋软掩码细化", True),
    ("p2_a4_mwf.json", "＋多通道维纳", True),
    ("p8_student.json", "蒸馏学生（3.1 M）", True),
    ("p1_oracle.json", "IRM oracle（掩码类上界）", False),
]


def main() -> int:
    p = argparse.ArgumentParser(description="SDR–RTF 帕累托曲线")
    p.add_argument("--out", default="results/p2_pareto.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.viz.fonts import use_cjk_font

    use_cjk_font(strict=True)

    rows = []
    for f, label, tunable in POINTS:
        d = json.loads((R / f).read_text(encoding="utf-8")) if (R / f).exists() else None
        if not d or "cSDR" not in d.get("median", {}):
            continue
        rtf = d.get("rtf_median")
        if rtf is None:
            continue
        rows.append({"label": label, "sdr": d["median"]["cSDR"]["mean"],
                     "rtf": rtf, "tunable": tunable})
    if not rows:
        print("❌ 没有可用数据点")
        return 1

    # 帕累托前沿：在**可选设置**里，没有别的点同时更快且更准
    tun = [r for r in rows if r["tunable"]]
    front = [r for r in tun
             if not any(o["rtf"] <= r["rtf"] and o["sdr"] >= r["sdr"] and o is not r
                        for o in tun)]
    front.sort(key=lambda r: r["rtf"])

    # trivial 在 −5.34 dB，把它画进来会把 7~9 dB 这个真正有意思的区间压成一条线。
    # 帕累托图的意义是看清取舍 —— 为了"完整"而牺牲可读性是本末倒置。
    # 它的值写在角注里，信息不丢。
    lo_pt = next((r for r in rows if "trivial" in r["label"]), None)
    rows = [r for r in rows if r is not lo_pt]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot([r["rtf"] for r in front], [r["sdr"] for r in front],
            "-", color="#4cc9f0", lw=1.6, zorder=1, label="帕累托前沿")
    # 标注位置逐点指定，不用统一偏移 —— 统一偏移在密集区必然重叠，
    # 第一版三个标签叠成一团，图上什么也读不出来。
    OFFSET = {
        "htdemucs 基线": (-10, -20),
        "＋overlap 0.50": (6, 12),
        "＋TTA": (-14, -20),
        "＋软掩码细化": (8, -14),
        "＋多通道维纳": (8, 8),
        "IRM oracle（掩码类上界）": (-4, 12),
        "蒸馏学生（3.1 M）": (12, 10),
    }
    for r in rows:
        on = r in front
        ax.scatter(r["rtf"], r["sdr"], s=120 if on else 75,
                   c="#4cc9f0" if on else ("#8b96a8" if r["tunable"] else "#d9a441"),
                   marker="o" if r["tunable"] else "^",
                   edgecolors="#1b2430", linewidths=1.2, zorder=3)
        ax.annotate(r["label"], (r["rtf"], r["sdr"]),
                    textcoords="offset points",
                    xytext=OFFSET.get(r["label"], (9, 6)), fontsize=9,
                    color="#1b2430" if on else "#5b6673")

    sdrs = [r["sdr"] for r in rows]
    rtfs = [r["rtf"] for r in rows]
    ax.set_ylim(min(sdrs) - 0.55, max(sdrs) + 0.45)
    ax.set_xlim(min(rtfs) - 0.012, max(rtfs) + 0.028)   # 右侧留白，否则 TTA 标签被切
    if lo_pt:
        ax.text(0.42, 0.045,
                f"下界 {lo_pt['label']} = {lo_pt['sdr']:.2f} dB（超出纵轴范围，未画）",
                transform=ax.transAxes, fontsize=8.5, color="#8b96a8")
    ax.set_xlabel("RTF（越小越快；0.066 ≈ 15× 实时）")
    ax.set_ylabel("cSDR 平均 (dB)")
    ax.set_title("分离质量 vs 推理速度：推理期能做的选择", fontsize=12)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out = ROOT / args.out
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"{'配置':<26}{'cSDR':>8}{'RTF':>9}{'前沿':>6}")
    for r in sorted(rows, key=lambda r: r["rtf"]):
        print(f"  {r['label']:<24}{r['sdr']:>8.2f}{r['rtf']:>9.4f}"
              f"{'  ✅' if r in front else ''}")
    base = next((r for r in rows if "基线" in r["label"]), None)
    tta = next((r for r in rows if "TTA" in r["label"]), None)
    if base and tta:
        # 质量差用**配对检验**的结果，不能拿两个聚合均值相减。
        # cSDR 是"每首每轨取分块中位数"再聚合，中位数不是线性算子，
        # mean(A)−mean(B) ≠ mean(A−B)：聚合相减是 +0.14 dB，配对检验是 +0.095 dB。
        # 速度可以直接相除（RTF 是均值，比值有意义）。
        from scripts.build_summary import parse_paired_ablation

        paired = {name: delta for name, delta, *_ in parse_paired_ablation()}
        d = paired.get("＋TTA(swap,flip)", "?")
        print(f"\n  TTA 的代价：RTF ×{tta['rtf']/base['rtf']:.1f} 换 {d}（配对检验）")
        print(f"  注意：图上纵轴是聚合后的 cSDR，两点之差（"
              f"{tta['sdr']-base['sdr']:+.2f} dB）**不等于**配对检验的结果。")
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
