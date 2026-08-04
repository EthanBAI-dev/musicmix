"""把多种子的训练结果汇总成 L0→L4 阶梯表 + 配置间的配对比较。

    python -m scripts.build_tagging_ladder --seeds-dir results/seeds --notes results/P4_分析.md

为什么必须多种子：训练集只有一千多首、测试集几百首，**单次运行的 0.005 mAP
差异完全可能是随机种子造成的**。P1/P2 已经吃过"从单个数字读结论"的亏，
这里从一开始就按同一套纪律做：报 mean±std，并对**相同种子**配对比较，
看差值的方向是否一致（5/5 一致才算数，2/5 就是噪声）。
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
from pathlib import Path

import numpy as np

METRICS = [
    ("roc_auc", "ROC-AUC"),
    ("map", "mAP"),
    ("macro_f1_default", "MacroF1@0.5"),
    ("macro_f1_tuned", "MacroF1@tuned"),
    ("micro_f1_tuned", "MicroF1@tuned"),
]


def load_seeds(pattern: str) -> dict[str, dict[str, list[float]]]:
    out: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(pattern)):
        m = re.match(r".*/(.+)_s(\d+)\.json$", f)
        if not m:
            continue
        name = m.group(1)
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if not d.get("test"):
            continue
        for k, _ in METRICS:
            out[name][k].append(d["test"][k])
        for g, gs in (d["test"].get("groups") or {}).items():
            out[name][f"group::{g}::map"].append(gs["map"])
            out[name][f"group::{g}::macro_f1_tuned"].append(gs["macro_f1_tuned"])
        out[name]["_params"].append(d.get("n_params", 0))
    return out


def ms(x: list[float]) -> str:
    a = np.asarray(x, dtype=float)
    return f"{a.mean():.4f}±{a.std(ddof=1):.4f}" if len(a) > 1 else f"{a.mean():.4f}"


def paired(rows, a: str, b: str, metric: str):
    """对**相同种子**配对求差。返回 (均值, 标准差, b 更好的次数, 总次数)。"""
    x, y = np.array(rows[a][metric]), np.array(rows[b][metric])
    n = min(len(x), len(y))
    d = y[:n] - x[:n]
    return d.mean(), d.std(ddof=1) if n > 1 else 0.0, int((d > 0).sum()), n


def main() -> int:
    p = argparse.ArgumentParser(description="汇总标签阶梯表")
    p.add_argument("--seeds-dir", default="results/seeds")
    p.add_argument("--single", nargs="*", default=["results/p4_l0.json"],
                   help="只跑了一个种子的结果（如 L0），单独列出")
    p.add_argument("--compare", action="append", default=[],
                   metavar="A=B=说明", help="配对比较，如 L2mean=L2attn=注意力−平均池化")
    p.add_argument("--notes", default="")
    p.add_argument("--out", default="results/P4_标签阶梯.md")
    args = p.parse_args()

    rows = load_seeds(f"{args.seeds_dir}/*.json")
    for f in args.single:
        path = Path(f)
        if not path.exists():
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("test"):
            name = d["level"]
            for k, _ in METRICS:
                rows[name][k].append(d["test"][k])
            for g, gs in (d["test"].get("groups") or {}).items():
                rows[name][f"group::{g}::map"].append(gs["map"])
                rows[name][f"group::{g}::macro_f1_tuned"].append(gs["macro_f1_tuned"])
            rows[name]["_params"].append(d.get("n_params", 0))

    if not rows:
        print("❌ 没有任何结果")
        return 1

    lines = [
        "# P4 标签阶梯（MTG-Jamendo top50tags）",
        "",
        "- 指标全部在 **test** 上；阈值在验证集搜好后固定，**绝不在测试集重搜**",
        "- 多种子的报 `mean±std`（ddof=1）",
        "- **mAP 与阈值无关**，所以 `MacroF1@0.5` 与 `MacroF1@tuned` 两列之间 mAP 不变",
        "",
        "## 主表",
        "",
        "| 配置 | 种子数 | 参数量 | " + " | ".join(n for _, n in METRICS) + " |",
        "|---" * (len(METRICS) + 3) + "|",
    ]
    for name in rows:
        r = rows[name]
        params = int(np.mean(r["_params"])) if r.get("_params") else 0
        lines.append(f"| {name} | {len(r['map'])} | {params:,} | "
                     + " | ".join(ms(r[k]) for k, _ in METRICS) + " |")

    # 分类别
    groups = sorted({k.split("::")[1] for r in rows.values() for k in r if k.startswith("group::")})
    if groups:
        lines += ["", "## 分类别 mAP（test）", "",
                  "| 配置 | " + " | ".join(groups) + " |", "|---" * (len(groups) + 1) + "|"]
        for name, r in rows.items():
            cells = [ms(r[f"group::{g}::map"]) if f"group::{g}::map" in r else "—" for g in groups]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")

    # 配对比较
    if args.compare:
        lines += ["", "## 配对比较（相同种子）", "",
                  "同一个种子跑两个配置再求差，能消掉初始化带来的共同波动。",
                  "**方向一致（5/5）才算数，2/5 就是噪声。**", "",
                  "| 比较 | 指标 | Δ | 方向一致性 | 判定 |", "|---|---|---|---|---|"]
        for spec in args.compare:
            a, b, desc = spec.split("=", 2)
            if a not in rows or b not in rows:
                print(f"⚠️  跳过 {spec}（缺 {a} 或 {b}）")
                continue
            for metric, mname in (("map", "mAP"), ("macro_f1_tuned", "MacroF1@tuned")):
                mu, sd, w, n = paired(rows, a, b, metric)
                verdict = ("✅ 一致更好" if w == n else
                           "❌ 一致更差" if w == 0 else "⚠️ 与噪声不可区分")
                lines.append(f"| {desc} | {mname} | {mu:+.4f}±{sd:.4f} | {w}/{n} | {verdict} |")

    if args.notes and Path(args.notes).exists():
        lines += ["", "---", "", Path(args.notes).read_text(encoding="utf-8").rstrip()]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: 12 + len(rows)]))
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
