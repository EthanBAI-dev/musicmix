"""音源分离评测的命令行入口。

参照基线（理论值已知，用来验证评测代码本身）：

- ``silence``  输出全零 → uSDR 恒为 0.00 dB，最硬的自检
- ``trivial``  混音当每一轨 → **下界锚点**
- ``oracle``   IRM 理想比值掩码 → **上界锚点**

真实模型（demucs 权重名）：``htdemucs`` / ``htdemucs_ft`` / ``mdx_extra`` …
**成绩必须落在 trivial 和 oracle 之间。落到外面就是哪里错了。**

用法::

    # 不需要数据集，验证评测框架
    python -m scripts.run_separation_eval --synthetic 8 --model oracle

    # 真实数据
    python -m scripts.run_separation_eval --model trivial  --subset test --workers 6 --out results/p1_trivial.md
    python -m scripts.run_separation_eval --model htdemucs --subset test --workers 3 --out results/p1_htdemucs.md

关于 ``--workers``：museval 是整个流程的瓶颈（一首 4 分钟的歌约 60 秒），
且是单线程 CPU 密集。并行化收益极大（M2 Max 上 6 进程约快 5 倍）。
跑 GPU 模型时每个子进程会各自加载一份权重，所以 workers 不宜开太大。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from src.audio.io import match_length
from src.bench.timer import device_info
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

REFERENCE_MODELS = ("silence", "trivial", "oracle")


# --------------------------------------------------------------------------------------
# 配置：必须可 pickle，因为 macOS 上 multiprocessing 用 spawn
# --------------------------------------------------------------------------------------

@dataclass
class EvalConfig:
    model: str
    # --model student 时指向权重文件；其他模型忽略
    ckpt: str = ""
    device: str = "auto"
    overlap: float = 0.25
    shifts: int = 0
    compute_csdr: bool = True
    save_stems: str = ""
    root: str = "data/musdb18hq"
    subset: str = "test"
    synthetic_seconds: float = 6.0
    # --- P2 路线 A：推理期增益，各自独立开关，方便做逐项累加的消融 ---
    tta: tuple[str, ...] = ()          # 空 = 不做 TTA
    refine: str = ""                   # "" | "mask" | "mwf"
    refine_alpha: float = 2.0
    ensemble_models: tuple[str, ...] = ()   # 非空则忽略 model，改用这组模型集成


# --------------------------------------------------------------------------------------
# 估计器
# --------------------------------------------------------------------------------------

_SEPARATOR = None       # 每个进程各自缓存一份，避免每首歌重新加载权重


def build_separator(cfg: EvalConfig):
    """返回 ``(fn, label)``；``fn(mixture, references) -> {stem: (n,2)}``。

    真实模型的调用链（每一层都可单独关掉，这正是消融表的结构）：
    ``模型推理 → [多模型集成] → [TTA 平均] → [软掩码 / MWF 细化]``
    """
    if cfg.model == "silence":
        return (lambda mix, refs: silence_baseline(mix)), "silence（全零，自检）"
    if cfg.model == "trivial":
        return (lambda mix, refs: trivial_baseline(mix)), "trivial（混音当每一轨，下界）"
    if cfg.model == "oracle":
        def _oracle(mix, refs):
            est = ideal_ratio_mask(refs, mix)
            return {k: match_length(v, mix.shape[0]) for k, v in est.items()}
        return _oracle, "oracle（IRM 理想掩码，上界）"

    if cfg.model == "student":
        # 蒸馏出来的学生模型。走与 demucs 相同的评测路径，
        # 这样它在表里的数字与教师**完全可比** —— 换个脚本评就没法比了。
        if not cfg.ckpt:
            raise ValueError("--model student 需要 --ckpt 指向训练好的权重")
        import numpy as np
        import torch

        from src.separation.student import SOURCES, StudentUNet, separate_chunked
        from src.tagging.backbone import pick_device

        ck = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
        conf = ck.get("config", {})
        dev = pick_device(cfg.device)
        model = StudentUNet(**{k: v for k, v in conf.items() if k in ("base", "depth")})
        model.load_state_dict(ck["state_dict"])
        model.to(dev).eval()

        @torch.no_grad()
        def _student(mix, refs):
            # **必须分段推理**：整首一次前向会 OOM（实测 exit 137）。
            # 4 分钟的歌 STFT 后 513×41,400 帧，U-Net 第一层就要 2.7 GB。
            x = torch.from_numpy(np.ascontiguousarray(mix.T)).float()
            out = separate_chunked(model, x, sr=44100).numpy()   # (S, 2, n)
            return {name: out[i].T for i, name in enumerate(SOURCES)}

        n_p = model.n_params
        return _student, f"student（蒸馏，{n_p/1e6:.2f} M 参数，step {ck.get('step', '?')}）"

    from src.separation import demucs_model, postprocess

    names = cfg.ensemble_models or (cfg.model,)
    seps = [demucs_model.load(n, device=cfg.device, overlap=cfg.overlap, shifts=cfg.shifts)
            for n in names]

    def base(mix):
        outs = [demucs_model.separate(s, mix) for s in seps]
        return outs[0] if len(outs) == 1 else postprocess.ensemble(outs)

    def run(mix, refs):
        est = postprocess.apply_tta(base, mix, cfg.tta) if cfg.tta else base(mix)
        if cfg.refine == "mask":
            est = postprocess.soft_mask_refine(est, mix, alpha=cfg.refine_alpha)
        elif cfg.refine == "mwf":
            est = postprocess.multichannel_wiener(est, mix, alpha=cfg.refine_alpha)
        return est

    parts = ["+".join(names), f"overlap={cfg.overlap}"]
    if cfg.shifts:
        parts.append(f"shifts={cfg.shifts}")
    if cfg.tta:
        parts.append(f"TTA[{','.join(cfg.tta)}]")
    if cfg.refine:
        parts.append(f"refine={cfg.refine}(α={cfg.refine_alpha})")
    return run, f"{seps[0].device} · " + " · ".join(parts)


def get_separator(cfg: EvalConfig):
    global _SEPARATOR
    if _SEPARATOR is None:
        _SEPARATOR = build_separator(cfg)
    return _SEPARATOR


# --------------------------------------------------------------------------------------
# 单首歌：worker 函数必须在模块顶层才能被 spawn 的子进程 pickle
# --------------------------------------------------------------------------------------

def eval_one(payload: tuple[EvalConfig, str, object]) -> tuple[TrackScores, float, float]:
    cfg, kind, ident = payload

    if kind == "synthetic":
        from src.datasets.synthetic import SyntheticTrack
        track = SyntheticTrack(int(ident), cfg.synthetic_seconds)
    else:
        from src.datasets.musdb import Track
        name, path = ident
        track = Track(name=name, path=Path(path), subset=cfg.subset)

    refs = track.references()
    mix = track.mixture()
    audio_sec = mix.shape[0] / 44100.0

    separator, _ = get_separator(cfg)
    t0 = time.perf_counter()
    ests = separator(mix, refs)
    sep_s = time.perf_counter() - t0

    if cfg.save_stems:
        from src.audio.io import save_audio
        for name_, y in ests.items():
            save_audio(Path(cfg.save_stems) / track.name / f"{name_}.wav", y)

    scores = evaluate_track(refs, ests, track.name, compute_csdr=cfg.compute_csdr)
    return scores, sep_s, audio_sec


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
    # silence 基线按定义就是全零，而 BSS Eval v4 要求估计非全零（否则投影方程组欠定）。
    # 这是 BSS Eval 的固有约束，不是缺陷 —— 它的意义本来就只在 uSDR 那条"恰好 0.00 dB"上。
    compute_csdr = not args.no_csdr
    if args.model == "silence" and compute_csdr:
        print("ℹ️  silence 基线全零，BSS Eval 无法处理，自动跳过 cSDR（只看 uSDR 是否恰为 0.00 dB）")
        compute_csdr = False

    cfg = EvalConfig(
        model=args.model, ckpt=args.ckpt,
        device=args.device, overlap=args.overlap, shifts=args.shifts,
        compute_csdr=compute_csdr, save_stems=args.save_stems,
        root=args.root, subset=args.subset, synthetic_seconds=args.synthetic_seconds,
        tta=tuple(args.tta), refine=args.refine, refine_alpha=args.refine_alpha,
        ensemble_models=tuple(args.ensemble),
    )

    # ---- 组装任务列表 ----
    if args.synthetic:
        items = [(cfg, "synthetic", i) for i in range(args.synthetic)]
        names = [f"synthetic-{i:02d}" for i in range(args.synthetic)]
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
        items = [(cfg, "musdb", (t.name, str(t.path))) for t in tracks]
        names = [t.name for t in tracks]
        source = f"MUSDB18-HQ {args.subset}（{len(tracks)} 首）"

    _, model_label = build_separator(cfg) if args.workers == 1 else (None, _label_only(cfg))

    print(f"数据源：{source}")
    print(f"模型  ：{model_label}")
    print(f"cSDR  ：{'开启（museval，较慢）' if compute_csdr else '关闭（只算 uSDR / SI-SDR）'}")
    print(f"并行  ：{args.workers} 进程\n")

    scores: list[TrackScores] = []
    sep_times: list[tuple[float, float]] = []
    t_start = time.perf_counter()
    total = len(items)

    def report(i: int, s: TrackScores, sep_s: float, audio_sec: float) -> None:
        shown = s.csdr if compute_csdr else s.usdr
        vals = [v for v in shown.values() if np.isfinite(v)]
        mean = np.mean(vals) if vals else float("nan")
        rtf = sep_s / audio_sec if audio_sec else 0.0
        done = time.perf_counter() - t_start
        eta = done / i * (total - i)
        print(f"[{i:>3}/{total}] {s.track_name[:42]:<42} 平均 {mean:6.2f} dB "
              f"(分离 {sep_s:5.1f}s, RTF {rtf:.3f})  ETA {eta / 60:.1f}min", flush=True)

    if args.workers == 1:
        for i, item in enumerate(items, 1):
            s, sep_s, audio_sec = eval_one(item)
            scores.append(s); sep_times.append((sep_s, audio_sec))
            report(i, s, sep_s, audio_sec)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(eval_one, item): item for item in items}
            for i, fut in enumerate(as_completed(futures), 1):
                s, sep_s, audio_sec = fut.result()
                scores.append(s); sep_times.append((sep_s, audio_sec))
                report(i, s, sep_s, audio_sec)
        # 并行完成顺序是乱的，按原始曲目顺序排回来，保证 CSV 可复现
        order = {n: k for k, n in enumerate(names)}
        pairs = sorted(zip(scores, sep_times, strict=True), key=lambda p: order.get(p[0].track_name, 1e9))
        scores = [p[0] for p in pairs]
        sep_times = [p[1] for p in pairs]

    elapsed = time.perf_counter() - t_start
    total_audio = sum(a for _, a in sep_times)
    rtfs = sorted(s / a for s, a in sep_times if a)
    rtf_median = rtfs[len(rtfs) // 2] if rtfs else float("nan")

    # ---- 汇总 ----
    agg_median = aggregate(scores, agg="median")
    agg_mean = aggregate(scores, agg="mean")

    print("\n" + "=" * 78)
    if compute_csdr and "cSDR" in agg_median:
        print(format_table({args.model: agg_median["cSDR"]}, metric="cSDR（中位数聚合，museval 官方口径）"))
        print()
    print(format_table({args.model: agg_mean["uSDR"]}, metric="uSDR（均值聚合，MDX 口径）"))
    print("=" * 78)
    if np.isfinite(rtf_median) and rtf_median > 0:
        print(f"分离 RTF 中位数 {rtf_median:.4f}（×{1 / rtf_median:.1f} 实时）"
              f"　总耗时 {elapsed / 60:.1f} min")

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
            f"| 模型 | {model_label} |",
            f"| 日期 | {datetime.now():%Y-%m-%d %H:%M} |",
            f"| commit | `{git_commit()}` |",
            f"| 硬件 | {info.get('cpu', info.get('machine'))} / torch {info.get('torch', '?')} |",
            f"| 命令 | `{' '.join(sys.argv)}` |",
            f"| 分离 RTF（中位数） | "
            + (f"{rtf_median:.4f}（×{1 / rtf_median:.1f} 实时）"
               if np.isfinite(rtf_median) and rtf_median > 0 else "—") + " |",
            f"| 总耗时 | {elapsed / 60:.1f} min（音频总长 {total_audio / 60:.1f} min，含 museval，{args.workers} 进程） |",
            "",
        ]
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
        import pandas as pd
        csv_path = out.with_suffix(".csv")
        pd.DataFrame([s.to_row() for s in scores]).to_csv(csv_path, index=False)
        print(f"✅ 逐首数据 → {csv_path}")

        json_path = out.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {"model": args.model, "label": model_label, "source": source,
                 "rtf_median": rtf_median if np.isfinite(rtf_median) else None,
                 "median": agg_median, "mean": agg_mean},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"✅ 机器可读 → {json_path}")

    return 0


def _label_only(cfg: EvalConfig) -> str:
    """并行模式下主进程不加载模型（免得白占一份显存），只生成标签。

    必须和 :func:`build_separator` 里的标签保持一致，否则结果表会标错配置 ——
    消融表里标错配置比数字算错更难发现。
    """
    if cfg.model in REFERENCE_MODELS:
        return {"silence": "silence（全零，自检）",
                "trivial": "trivial（混音当每一轨，下界）",
                "oracle": "oracle（IRM 理想掩码，上界）"}[cfg.model]
    parts = ["+".join(cfg.ensemble_models or (cfg.model,)), f"overlap={cfg.overlap}"]
    if cfg.shifts:
        parts.append(f"shifts={cfg.shifts}")
    if cfg.tta:
        parts.append(f"TTA[{','.join(cfg.tta)}]")
    if cfg.refine:
        parts.append(f"refine={cfg.refine}(α={cfg.refine_alpha})")
    return f"{cfg.device} · " + " · ".join(parts)


def main() -> int:
    p = argparse.ArgumentParser(
        description="音源分离评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model", default="trivial",
                   help="参照基线 silence/trivial/oracle，"
                        "蒸馏学生 student（配 --ckpt），"
                        "或 demucs 权重名 htdemucs / htdemucs_ft / mdx_extra …")
    p.add_argument("--ckpt", default="",
                   help="--model student 时指向蒸馏出来的权重（.pt）")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    p.add_argument("--overlap", type=float, default=0.25,
                   help="分块推理重叠率。官方默认 0.25，**做 baseline 不要改**")
    p.add_argument("--shifts", type=int, default=0,
                   help="随机时移 TTA 次数。>0 就不是 baseline 了（属 P2-A）")
    p.add_argument("--tta", nargs="*", default=[],
                   choices=("identity", "swap", "flip", "swap_flip"),
                   help="测试时增强的变换列表。推理次数 = 变换数，RTF 线性增长")
    p.add_argument("--refine", default="", choices=("", "mask", "mwf"),
                   help="后处理：mask=软掩码(α-Wiener)，mwf=多通道维纳滤波")
    p.add_argument("--refine-alpha", type=float, default=2.0)
    p.add_argument("--ensemble", nargs="*", default=[],
                   help="多模型集成，如 --ensemble htdemucs htdemucs_ft mdx_extra（会忽略 --model）")
    p.add_argument("--workers", type=int, default=1,
                   help="并行进程数。museval 是瓶颈，参照基线可开 6；GPU 模型建议 2~3")
    p.add_argument("--save-stems", default="", metavar="DIR",
                   help="把分离结果存到该目录（供前端试听 / 失败案例分析）")
    p.add_argument("--root", default="data/musdb18hq", help="MUSDB18-HQ 根目录")
    p.add_argument("--subset", default="test", choices=("train", "test"))
    p.add_argument("--limit", type=int, default=0, help="只评前 N 首（调试用）")
    p.add_argument("--synthetic", type=int, default=0, metavar="N",
                   help="不用真实数据，改用 N 首合成歌曲验证评测框架")
    p.add_argument("--synthetic-seconds", type=float, default=6.0)
    p.add_argument("--no-csdr", action="store_true", help="跳过 museval（慢），只算 uSDR / SI-SDR")
    p.add_argument("--out", default="", help="输出 markdown 路径，同时写同名 .csv 和 .json")
    args = p.parse_args()

    # 每个子进程内部再开多线程会互相抢核，反而更慢
    if args.workers > 1:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ.setdefault(var, "2")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
