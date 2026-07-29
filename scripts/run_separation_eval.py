"""音源分离评测的命令行入口。

M0 验收：**在还没下载 MUSDB18-HQ 之前，用合成数据就能证明评测框架是对的。**

    # 合成数据自检（不需要任何数据集，30 秒跑完）
    python -m scripts.run_separation_eval --synthetic 8

    # 真实数据（需要 data/musdb18hq/）
    python -m scripts.run_separation_eval --model trivial --subset test
    python -m scripts.run_separation_eval --model oracle  --subset test --limit 5

模型（``--model``）目前只有三个"参照物"，真实模型在 P1 接进来：

- ``silence``  输出全零 → uSDR 恒为 0.00 dB，最硬的自检
- ``trivial``  混音当每一轨 → **下界锚点**
- ``oracle``   IRM 理想比值掩码 → **上界锚点**

真实模型的成绩必须落在 trivial 和 oracle 之间。落到外面就是哪里错了。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.audio.io import match_length
from src.eval.separation import (
    STEMS,
    TrackScores,
    aggregate,
    evaluate_track,
    format_table,
    ideal_ratio_mask,
    silence_baseline,
    trivial_baseline,
)
from src.bench.timer import device_info
from src.datasets.synthetic import SyntheticTrack

MODELS = ("silence", "trivial", "oracle")


# --------------------------------------------------------------------------------------
# 估计器
# --------------------------------------------------------------------------------------

def separate(model: str, mixture: np.ndarray, references: dict) -> dict[str, np.ndarray]:
    """按名字产生一组估计。P1 接入真实模型时在这里加分支。"""
    if model == "silence":
        return silence_baseline(mixture)
    if model == "trivial":
        return trivial_baseline(mixture)
    if model == "oracle":
        est = ideal_ratio_mask(references, mixture)
        n = mixture.shape[0]
        return {k: match_length(v, n) for k, v in est.items()}
    raise ValueError(f"未知模型 {model!r}，可选：{MODELS}")


# --------------------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------------------

def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        ).stdout.strip() or "(未提交)"
    except Exception:
        return "(未知)"


def run(args: argparse.Namespace) -> int:
    if args.synthetic:
        tracks = [SyntheticTrack(i, args.synthetic_seconds) for i in range(args.synthetic)]
        source = f"合成数据 ×{args.synthetic}（每首 {args.synthetic_seconds}s）"
    else:
        from src.datasets.musdb import load_tracks

        try:
            tracks = load_tracks(args.root, args.subset)
        except FileNotFoundError as e:
            print(f"❌ {e}", file=sys.stderr)
            print("\n提示：想先验证评测框架本身，用 --synthetic 8", file=sys.stderr)
            return 1
        if args.limit:
            tracks = tracks[: args.limit]
        source = f"MUSDB18-HQ {args.subset}（{len(tracks)} 首）"

    # silence 基线按定义就是全零，而 BSS Eval v4 要求估计非全零（否则投影方程组欠定）。
    # 这是 BSS Eval 的固有约束，不是缺陷 —— 它的意义本来就只在 uSDR 那条"恰好 0.00 dB"上。
    compute_csdr = not args.no_csdr
    if args.model == "silence" and compute_csdr:
        print("ℹ️  silence 基线全零，BSS Eval 无法处理，自动跳过 cSDR（只看 uSDR 是否恰为 0.00 dB）")
        compute_csdr = False

    print(f"数据源：{source}")
    print(f"模型  ：{args.model}")
    print(f"cSDR  ：{'开启（museval，较慢）' if compute_csdr else '关闭（只算 uSDR / SI-SDR）'}\n")

    scores: list[TrackScores] = []
    t_start = time.perf_counter()

    for i, track in enumerate(tracks, 1):
        refs = track.references()
        mix = track.mixture()

        t0 = time.perf_counter()
        ests = separate(args.model, mix, refs)
        sep_s = time.perf_counter() - t0

        s = evaluate_track(refs, ests, track.name, compute_csdr=compute_csdr)
        scores.append(s)

        shown = s.csdr if compute_csdr else s.usdr
        mean = np.mean([v for v in shown.values() if np.isfinite(v)])
        print(
            f"[{i:>3}/{len(tracks)}] {track.name[:46]:<46} "
            f"平均 {mean:6.2f} dB  (分离 {sep_s:.1f}s)"
        )

    elapsed = time.perf_counter() - t_start
    total_audio = sum(getattr(t, "duration_seconds", 0.0) for t in tracks)

    # ---- 汇总 ----
    agg_median = aggregate(scores, agg="median")
    agg_mean = aggregate(scores, agg="mean")

    print("\n" + "=" * 78)
    if compute_csdr:
        print(format_table({args.model: agg_median["cSDR"]}, metric="cSDR（中位数聚合，museval 官方口径）"))
        print()
    print(format_table({args.model: agg_mean["uSDR"]}, metric="uSDR（均值聚合，MDX 口径）"))
    print("=" * 78)

    # ---- 落盘 ----
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        info = device_info()

        lines = [
            f"# 分离评测结果：{args.model}",
            "",
            "| 项 | 值 |",
            "|---|---|",
            f"| 数据源 | {source} |",
            f"| 模型 | `{args.model}` |",
            f"| 日期 | {datetime.now():%Y-%m-%d %H:%M} |",
            f"| commit | `{git_commit()}` |",
            f"| 硬件 | {info.get('cpu', info.get('machine'))} / {info.get('device')} |",
            f"| 命令 | `{' '.join(sys.argv)}` |",
            f"| 总耗时 | {elapsed:.1f}s（音频总长 {total_audio:.0f}s） |",
            "",
        ]
        # 某个指标可能整体算不出来（比如 silence 基线的 SI-SDR 全是 -inf），
        # 这时它不会出现在聚合结果里。缺就跳过并注明，不要崩，也不要假装有值。
        if compute_csdr:
            for m in ("cSDR", "ISR", "SIR", "SAR"):
                if m in agg_median:
                    lines += [format_table({args.model: agg_median[m]}, metric=f"{m}（中位数聚合）"), ""]
        for m in ("uSDR", "SI-SDR"):
            if m in agg_mean:
                lines += [format_table({args.model: agg_mean[m]}, metric=f"{m}（均值聚合）"), ""]
            else:
                lines += [f"> ⚠️ {m}：全部曲目均为非有限值，无法聚合（该基线下属预期行为）。", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"\n✅ 汇总表 → {out}")

        # 逐首原始数据：显著性检验和失败案例定位都要用它，比汇总表重要
        csv_path = out.with_suffix(".csv")
        import pandas as pd

        pd.DataFrame([s.to_row() for s in scores]).to_csv(csv_path, index=False)
        print(f"✅ 逐首数据 → {csv_path}")

        json_path = out.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {"model": args.model, "source": source, "median": agg_median, "mean": agg_mean},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"✅ 机器可读 → {json_path}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="音源分离评测（M0 参照基线）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="trivial", choices=MODELS)
    p.add_argument("--root", default="data/musdb18hq", help="MUSDB18-HQ 根目录")
    p.add_argument("--subset", default="test", choices=("train", "test"))
    p.add_argument("--limit", type=int, default=0, help="只评前 N 首（调试用）")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="不用真实数据，改用 N 首合成歌曲验证评测框架")
    p.add_argument("--synthetic-seconds", type=float, default=6.0)
    p.add_argument("--no-csdr", action="store_true",
                   help="跳过 museval（慢），只算 uSDR / SI-SDR")
    p.add_argument("--out", default="", help="输出 markdown 路径，同时写同名 .csv 和 .json")
    return run(p.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
