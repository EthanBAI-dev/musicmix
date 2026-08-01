"""把多次评测的逐首结果汇总成消融表 + 配对显著性检验。

    python -m scripts.build_ablation_table \\
        --base results/p1_htdemucs.csv \\
        --variant "＋overlap 0.50=results/p2_a1_overlap50.csv" \\
        --variant "＋TTA(swap,flip)=results/p2_a2_tta.csv" \\
        --out results/P2_消融实验.md

为什么要单独一个脚本：消融表是 P2 的**核心交付物**，必须能一条命令重跑。
手工从各个 md 里抄数字，抄错一个就全废，而且没人能复现。

每个变体都对 base 做配对 bootstrap（n=50，对同样的 50 首歌），
因为 P1 量到曲间标准差 2.25 dB —— **只报均值差是站不住的**。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.stats import compare_models

STEMS = ("vocals", "drums", "bass", "other")


def load(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv).set_index("track")
    missing = [c for c in (f"cSDR_{s}" for s in STEMS) if c not in df.columns]
    if missing:
        raise ValueError(f"{csv} 缺少列：{missing}")
    return df


def rtf_of(csv: Path) -> float | None:
    """从同名 json 里取分离 RTF —— 消融表必须把代价一起报，不然只看收益是耍流氓。"""
    j = csv.with_suffix(".json")
    if not j.exists():
        return None
    return json.loads(j.read_text(encoding="utf-8")).get("rtf_median")


def main() -> int:
    p = argparse.ArgumentParser(description="生成分离消融表")
    p.add_argument("--base", required=True, help="基线的逐首 csv")
    p.add_argument("--base-name", default="base（htdemucs, overlap=0.25）")
    p.add_argument("--variant", action="append", default=[], metavar="名称=csv路径")
    p.add_argument("--out", default="results/P2_消融实验.md")
    p.add_argument("--title", default="P2 路线 A：推理期增益消融")
    args = p.parse_args()

    base_csv = Path(args.base)
    base = load(base_csv)
    cols = [f"cSDR_{s}" for s in STEMS]

    variants = []
    for spec in args.variant:
        name, _, path = spec.partition("=")
        csv = Path(path)
        if not csv.exists():
            print(f"⚠️  跳过 {name}：{csv} 不存在")
            continue
        df = load(csv)
        common = base.index.intersection(df.index)
        if len(common) != len(base):
            print(f"⚠️  {name} 与基线的曲目集合不一致（交集 {len(common)}/{len(base)}），只用交集")
        variants.append((name, df, csv, common))

    if not variants:
        print("❌ 没有可对比的变体")
        return 1

    lines = [
        f"# {args.title}",
        "",
        f"- 数据：MUSDB18-HQ test，{len(base)} 首",
        "- 指标：cSDR（museval / BSS Eval v4，1 秒分块）",
        "- **表内数值为 50 首的中位数聚合**（museval 官方口径）；",
        "  **Δ 与显著性用配对 bootstrap 对均值做**（10000 次重采样，n=50）。",
        "  两种聚合可能给出不同结论，这是 P1 踩过的坑，所以两者都列。",
        "",
        "## 主表（cSDR，中位数聚合）",
        "",
        "| 配置 | vocals | drums | bass | other | 平均 | RTF |",
        "|---|---|---|---|---|---|---|",
    ]

    def med_row(name: str, df: pd.DataFrame, idx, csv: Path) -> str:
        vals = [df.loc[idx, f"cSDR_{s}"].median() for s in STEMS]
        rtf = rtf_of(csv)
        rtf_s = f"{rtf:.4f}" if rtf else "—"
        return f"| {name} | " + " | ".join(f"{v:.2f}" for v in vals) \
               + f" | **{np.mean(vals):.2f}** | {rtf_s} |"

    lines.append(med_row(args.base_name, base, base.index, base_csv))
    for name, df, csv, common in variants:
        lines.append(med_row(name, df, common, csv))

    # ---- 逐变体的配对检验 ----
    lines += ["", "## 相对基线的配对检验（对均值，10000 次 bootstrap）", ""]
    summary = []
    for name, df, csv, common in variants:
        B = base.loc[common, cols].values
        V = df.loc[common, cols].values
        res = compare_models(B, V)
        lines += [f"### {name}", "", "| 声部 | Δ | 95% CI | p | 显著 |", "|---|---|---|---|---|"]
        for stem, r in res.items():
            lines.append(
                f"| {stem} | {r.mean_diff:+.3f} dB | "
                f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.p_value:.4f} | "
                f"{'✅' if r.significant else '❌'} |"
            )
        lines.append("")
        summary.append((name, res))

    # ---- 一页纸结论 ----
    lines += ["## 结论速览", "",
              "| 配置 | 平均 Δ | 显著 | 哪些声部显著变好 | 哪些显著变差 |",
              "|---|---|---|---|---|"]
    for name, res in summary:
        up = [s for s in STEMS if res[s].significant and res[s].mean_diff > 0]
        down = [s for s in STEMS if res[s].significant and res[s].mean_diff < 0]
        m = res["mean"]
        lines.append(
            f"| {name} | {m.mean_diff:+.3f} dB | {'✅' if m.significant else '❌'} | "
            f"{'、'.join(up) or '—'} | {'、'.join(down) or '—'} |"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-len(summary) - 4:]))
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
