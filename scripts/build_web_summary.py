"""把多种子实验结果汇总成前端能直接读的一个 JSON。

    python -m scripts.build_web_summary

前端不能 glob `results/seeds/*.json`，所以由这里生成 `results/web_summary.json`。

**这个文件只做汇总，不做判断的美化** —— 被证伪的自研点在页面上
仍然显示为「证伪」。作品集的价值在于把负结果也摆出来，
一个只列成功项的页面等于没说实话。
"""

from __future__ import annotations

import collections
import glob
import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# (键, 展示名, 分组) —— 分组决定在页面上排在哪一块
LADDER = [
    ("L0",       "mel + 小 CNN（从头训）",        "阶梯"),
    ("L1",       "MERT 线性探针",                 "阶梯"),
    ("L2mean",   "MERT + MLP 头（mean 池化）",    "阶梯"),
    ("M330L2",   "MERT-330M + MLP 头",            "阶梯"),
    ("SEGall4",  "MERT-95M + 全曲 4 段 ★",        "阶梯"),
    ("M330SEG4", "MERT-330M + 全曲 4 段",         "阶梯"),
]

# (基线, 处理, 说明, 这是不是"我自己设计的点")
COMPARISONS = [
    ("L2mean",  "L2attn",   "注意力池化 vs 平均池化",        True),
    ("L2mean",  "L2max",    "最大池化 vs 平均池化",          True),
    ("L2mean",  "L3focal",  "Focal 损失 vs BCE",             True),
    ("L2mean",  "L3asl",    "非对称损失 ASL vs BCE",         True),
    ("L2mean",  "L5gate",   "Stem-aware 融合（门控）",       True),
    ("L2mean",  "M330L2",   "换 330M 基座",                  False),
    ("SEGs2",   "SEGall4",  "全曲 4 段 vs 单段 30 秒 ★",     True),
    ("M330L2",  "SEGall4",  "4 段 95M vs 330M（参数少 3.35×）", False),
    ("SEGall4",  "SEGall8",  "8 段 vs 4 段（饱和点）★",        True),
    ("SEGall4",  "M330SEG4", "有 4 段后再换 330M",             False),
    ("SEGall4",  "A4attn",   "注意力池化（在 4 段输入上复检）", True),
    ("SEGall4",  "A4linear", "换回线性探针（在 4 段输入上）",   True),
]


def load_seeds() -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = collections.defaultdict(dict)
    for f in sorted(glob.glob(str(ROOT / "results/seeds/*.json"))):
        m = re.match(r".*/(.+)_s(\d+)\.json$", f)
        if not m:
            continue
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("test"):
            out[m.group(1)][int(m.group(2))] = d["test"]
    return out


def stat(runs: dict[int, dict], key: str) -> dict | None:
    if not runs:
        return None
    v = np.array([runs[s][key] for s in sorted(runs)])
    return {"mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "n": int(len(v))}


def compare(a: dict[int, dict], b: dict[int, dict], key: str = "map") -> dict | None:
    """同种子配对。只在**共同种子**上比 —— 拿不同种子的均值相减是没有意义的。"""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        return None
    d = np.array([b[s][key] - a[s][key] for s in seeds])
    wins = int((d > 0).sum())
    n = len(seeds)
    if wins == n:
        verdict = "confirmed"      # 方向一致更好
    elif wins == 0:
        verdict = "refuted"        # 方向一致更差
    else:
        verdict = "noise"          # 方向不一致 → 噪声
    return {"delta": float(d.mean()), "std": float(d.std(ddof=1)),
            "wins": wins, "n": n, "verdict": verdict}


def test_count() -> int:
    """从 pytest 实际收集数，不写死 —— 写死的数字迟早和仓库脱节。"""
    try:
        r = subprocess.run(
            ["/opt/anaconda3/envs/music-mix/bin/python", "-m", "pytest",
             "--collect-only", "-q", "-p", "no:cacheprovider", "tests/"],
            cwd=ROOT, capture_output=True, text=True, timeout=180)
        # `-q --collect-only` 每个文件输出一行 `tests/x.py: N`，末尾没有汇总行，
        # 所以按文件求和。别去匹配 "N tests collected" —— 这个版本不打印它。
        n = sum(int(m) for m in re.findall(r"^\S+\.py: (\d+)$", r.stdout, re.M))
        return n
    except Exception:
        return 0


def main() -> int:
    seeds = load_seeds()

    # L0 是单次运行（结果不在 seeds/ 里），单独读，并如实标注 n=1
    l0 = ROOT / "results/p4_l0.json"
    if l0.exists():
        d = json.loads(l0.read_text(encoding="utf-8")).get("test")
        if d:
            seeds["L0"] = {0: d}

    ladder = []
    for key, label, group in LADDER:
        runs = seeds.get(key, {})
        if not runs:
            continue
        ladder.append({"key": key, "label": label, "group": group,
                       "map": stat(runs, "map"),
                       "f1": stat(runs, "macro_f1_tuned")})

    comps = []
    for base, treat, label, is_mine in COMPARISONS:
        c = compare(seeds.get(base, {}), seeds.get(treat, {}))
        if c:
            comps.append({"label": label, "mine": is_mine, **c})

    out = {
        "generated_by": "python -m scripts.build_web_summary",
        "n_tests": test_count(),
        "ladder": ladder,
        "comparisons": comps,
        # 页面上要显示的一句话总结，直接由数据算出来，不手写
        "n_refuted": sum(1 for c in comps if c["mine"] and c["verdict"] == "refuted"),
        "n_noise": sum(1 for c in comps if c["mine"] and c["verdict"] == "noise"),
        "n_confirmed": sum(1 for c in comps if c["mine"] and c["verdict"] == "confirmed"),
    }
    p = ROOT / "results/web_summary.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"阶梯 {len(ladder)} 行，比较 {len(comps)} 项，单测 {out['n_tests']} 条")
    for c in comps:
        mark = {"confirmed": "✅", "refuted": "❌", "noise": "⚠️"}[c["verdict"]]
        print(f"  {mark} {c['label']:<34}Δ={c['delta']:+.4f}  {c['wins']}/{c['n']}")
    print(f"\n✅ → {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
