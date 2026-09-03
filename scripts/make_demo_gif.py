"""渲染一张演示 GIF：四轨波形 + 拍点/小节线网格 + 和弦 + 曲式 + 走带播放头。

    python -m scripts.make_demo_gif

**这是脚本生成的可视化，不是网页录屏。** 两者的区别必须说清楚：
录屏能证明"界面真的能用"，而这张图只能证明"分析结果长这样"。
把脚本渲染的图叫成"演示录屏"是不诚实的 —— 所以文件名与 README 里都注明来源。

画的内容与网页时间轴一致（同一份 analysis.json），
所以它如实反映了页面上会看到的信息，只是没有交互。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
STEMS = [("vocals", "#f4a261"), ("drums", "#e76f51"),
         ("bass", "#2a9d8f"), ("other", "#9b8ade")]
BG, PANEL, LINE, TEXT, DIM = "#0b0e14", "#141922", "#262e3b", "#e6ebf2", "#8b96a8"


def envelope(y: np.ndarray, n: int) -> np.ndarray:
    """把波形压成 n 个点的包络。取每段的最大绝对值，不是均值 ——
    均值会把瞬态抹平，画出来是一条没有起伏的香肠。"""
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return np.abs(y[: len(y) // n * n].reshape(n, -1)).max(axis=1)


def main() -> int:
    p = argparse.ArgumentParser(description="渲染演示 GIF")
    p.add_argument("--track", default="web/demo", help="含 mixture/stems + analysis.json 的目录")
    p.add_argument("--seconds", type=float, default=0.0, help="只画前 N 秒。0 = 全曲")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", default="docs/assets/demo.gif")
    args = p.parse_args()

    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.viz.fonts import use_cjk_font

    # strict=True：这张图要提交进仓库，中文变方框必须当场失败，
    # 而不是留一张"看起来生成成功"的坏图
    use_cjk_font(strict=True)
    from matplotlib.animation import FuncAnimation, PillowWriter

    d = ROOT / args.track
    aj = d / "analysis.json"
    if not aj.exists():
        print(f"❌ 缺少 {aj}。先跑 python -m scripts.backfill_analysis")
        return 1
    A = json.loads(aj.read_text(encoding="utf-8"))

    sr = 22050
    waves = {}
    for name, _ in STEMS:
        f = d / f"{name}.mp3"
        if f.exists():
            y, _ = librosa.load(str(f), sr=sr, mono=True)
            waves[name] = y
    if not waves:
        print(f"❌ {d} 里没有 stem 音频")
        return 1

    dur = min(len(y) for y in waves.values()) / sr
    if args.seconds:
        dur = min(dur, args.seconds)
    N = 900
    envs = {k: envelope(v[: int(dur * sr)], N) for k, v in waves.items()}
    t = np.linspace(0, dur, N)

    fig, axes = plt.subplots(len(envs) + 1, 1, figsize=(9, 5.2),
                             gridspec_kw={"height_ratios": [1.1] + [1] * len(envs)},
                             facecolor=BG)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.07, hspace=0.25)
    fig.suptitle(f"{A.get('source', '')}   {A['bpm']} BPM · {A['key']} · {A['time_signature']}"
                 f"   （脚本渲染，非网页录屏）",
                 color=TEXT, fontsize=11, y=0.975)

    # ---- 顶部：曲式 + 和弦 + 小节线 ----
    ax0 = axes[0]
    ax0.set_facecolor(PANEL)
    SEG_C = {"intro": "#2b3a4a", "outro": "#2b3a4a"}
    PALETTE = ["#1f3b52", "#3d2f52", "#1f4a3a", "#4a3520"]
    for i, s in enumerate(A.get("segments", [])):
        if s["end"] <= 0 or s["start"] >= dur:
            continue
        # 两种来源的 segments schema 不同：分析器输出带 cluster（同类段落同色），
        # 而演示曲的手写真值只有 label。缺 cluster 时退回按标签取色。
        cid = s.get("cluster", ord(s["label"][0]))
        c = SEG_C.get(s["label"], PALETTE[cid % len(PALETTE)])
        ax0.axvspan(max(s["start"], 0), min(s["end"], dur), color=c, lw=0)
        mid = (max(s["start"], 0) + min(s["end"], dur)) / 2
        if min(s["end"], dur) - max(s["start"], 0) > dur * 0.06:
            ax0.text(mid, 0.82, s["label"], color=TEXT, fontsize=8,
                     ha="center", va="center")
    for c in A.get("chords", []):
        if c["start"] < dur and c["end"] - c["start"] > dur * 0.03:
            ax0.text((c["start"] + min(c["end"], dur)) / 2, 0.42, c["label"],
                     color="#7ec8e3", fontsize=9, ha="center", va="center", weight="bold")
    for b in A.get("beats", []):
        if b <= dur:
            ax0.axvline(b, color=LINE, lw=0.6, ymin=0.02, ymax=0.18)
    for b in A.get("downbeats", []):
        if b <= dur:
            ax0.axvline(b, color=DIM, lw=1.4, ymin=0.02, ymax=0.28)
    ax0.set_xlim(0, dur)
    ax0.set_ylim(0, 1)
    ax0.set_yticks([])
    ax0.tick_params(colors=DIM, labelsize=8)
    for s in ax0.spines.values():
        s.set_color(LINE)

    # ---- 四轨波形 ----
    for ax, (name, color) in zip(axes[1:], [(k, dict(STEMS)[k]) for k in envs], strict=False):
        e = envs[name]
        ax.set_facecolor(PANEL)
        ax.fill_between(t, -e, e, color=color, lw=0, alpha=0.85)
        ax.set_xlim(0, dur)
        ax.set_ylim(-1.05, 1.05)
        ax.set_yticks([])
        ax.set_ylabel(name, color=color, fontsize=9, rotation=0,
                      ha="right", va="center", labelpad=18)
        ax.tick_params(colors=DIM, labelsize=8)
        for s in ax.spines.values():
            s.set_color(LINE)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel("秒", color=DIM, fontsize=9)

    heads = [ax.axvline(0, color="#4cc9f0", lw=1.6) for ax in axes]

    n_frames = int(dur * args.fps)

    def update(i):
        x = i / args.fps
        for h in heads:
            h.set_xdata([x, x])
        return heads

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, update, frames=n_frames, blit=True)
    anim.save(str(out), writer=PillowWriter(fps=args.fps), savefig_kwargs={"facecolor": BG})
    plt.close(fig)

    mb = out.stat().st_size / 2**20
    print(f"✅ → {out}  {n_frames} 帧 / {dur:.0f}s / {mb:.2f} MB")
    if mb > 5:
        print("⚠️ 超过 5 MB，GitHub 上加载会慢。用 --seconds 截短或降 --fps。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
