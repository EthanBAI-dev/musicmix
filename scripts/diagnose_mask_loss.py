"""诊断：软掩码细化的损失，到底来自幅度还是相位？

    python -m scripts.diagnose_mask_loss --tracks 10

P2 的消融显示软掩码细化让 htdemucs 净亏 1.36 dB。报告里给出的机制解释是
"掩码丢掉了 Demucs 重建的相位、被迫用回混音的相位"，但那只是**推测** ——
支撑它的是 P1 的 oracle 变体数据（真值幅度 + 混音相位只有 8.35 dB），
不是针对本次实验的隔离实验。这个脚本补上那一步。

把掩码这一步拆成两个可独立测量的因素：

===========================  ================================================
配置                          含义
===========================  ================================================
``base``                     模型输出：自己的幅度 + 自己的相位
``自幅度 + 混音相位``          只把相位换掉 → **单独隔离"丢弃相位"的代价**
``软掩码``                    幅度也重新归一化 → 完整的掩码投影
===========================  ================================================

如果 ``自幅度+混音相位`` 已经掉到接近 ``软掩码`` 的水平，
说明损失几乎全来自相位，掩码对幅度的那一步影响很小 —— 推测成立。
反之则要重新解释。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.audio.io import match_length
from src.eval.separation import STEMS, global_sdr

N_FFT, HOP = 4096, 1024


def _stft(x: np.ndarray) -> np.ndarray:
    import librosa
    return np.stack([librosa.stft(np.ascontiguousarray(x[:, c]), n_fft=N_FFT, hop_length=HOP)
                     for c in range(x.shape[1])])


def _istft(X: np.ndarray, n: int) -> np.ndarray:
    import librosa
    return np.stack([librosa.istft(X[c], hop_length=HOP, length=n) for c in range(X.shape[0])],
                    axis=1).astype(np.float32)


def swap_phase(estimates: dict[str, np.ndarray], mixture: np.ndarray) -> dict[str, np.ndarray]:
    """保留每个估计自己的幅度，只把相位换成混音的相位。

    这一步单独隔离了"掩码方法必然继承混音相位"这个约束的代价 ——
    幅度信息完全没动。
    """
    n = mixture.shape[0]
    phase = np.exp(1j * np.angle(_stft(mixture)))
    return {s: _istft(np.abs(_stft(match_length(y, n))) * phase, n)
            for s, y in estimates.items()}


def main() -> int:
    p = argparse.ArgumentParser(description="拆解软掩码细化的损失来源")
    p.add_argument("--tracks", type=int, default=10, help="用多少首（从 test 集等间隔抽样）")
    p.add_argument("--root", default="data/musdb18hq")
    p.add_argument("--out", default="results/p2_mask_loss_diagnosis.json")
    args = p.parse_args()

    from src.datasets.musdb import load_tracks
    from src.separation import demucs_model, postprocess

    all_tracks = load_tracks(args.root, "test")
    # 等间隔抽样而不是取前 N 首：MUSDB 的曲目顺序和风格有相关性，取前 N 会有偏
    step = max(1, len(all_tracks) // args.tracks)
    tracks = all_tracks[::step][: args.tracks]
    print(f"抽样 {len(tracks)} / {len(all_tracks)} 首（等间隔）")

    sep = demucs_model.load("htdemucs")
    configs = ("base", "自幅度+混音相位", "软掩码")
    acc = {c: {s: [] for s in STEMS} for c in configs}

    for i, t in enumerate(tracks, 1):
        mix, refs = t.mixture(), t.references()
        n = mix.shape[0]
        base = demucs_model.separate(sep, mix)
        variants = {
            "base": base,
            "自幅度+混音相位": swap_phase(base, mix),
            "软掩码": postprocess.soft_mask_refine(base, mix),
        }
        for c, est in variants.items():
            for s in STEMS:
                acc[c][s].append(global_sdr(refs[s], match_length(est[s], n)))
        print(f"[{i:>2}/{len(tracks)}] {t.name[:44]:<44} "
              + "  ".join(f"{c}={np.mean([acc[c][s][-1] for s in STEMS]):5.2f}" for c in configs),
              flush=True)

    means = {c: {s: float(np.mean(acc[c][s])) for s in STEMS} for c in configs}
    for c in configs:
        means[c]["mean"] = float(np.mean([means[c][s] for s in STEMS]))

    print("\n" + "=" * 76)
    print(f"{len(tracks)} 首 test 曲目，uSDR 均值（dB）")
    print("=" * 76)
    print(f"| {'配置':<16} | " + " | ".join(f"{s:>7}" for s in STEMS) + " |   平均 |")
    print("|" + "-" * 18 + "|" + ("-" * 9 + "|") * 4 + "-" * 8 + "|")
    for c in configs:
        print(f"| {c:<16} | " + " | ".join(f"{means[c][s]:7.2f}" for s in STEMS)
              + f" | {means[c]['mean']:6.2f} |")

    d_phase = means["自幅度+混音相位"]["mean"] - means["base"]["mean"]
    d_mag = means["软掩码"]["mean"] - means["自幅度+混音相位"]["mean"]
    total = means["软掩码"]["mean"] - means["base"]["mean"]
    share = d_phase / total * 100 if total else float("nan")

    print("\n损失拆解（相对 base）")
    print(f"  ① 只换相位（幅度不动）      Δ = {d_phase:+6.2f} dB")
    print(f"  ② 再让掩码重新分配幅度      Δ = {d_mag:+6.2f} dB")
    print(f"  合计（= 软掩码）            Δ = {total:+6.2f} dB")
    print(f"\n→ 相位这一步占总损失的 {share:.0f}%")
    print("→ 结论：" + ("推测成立，损失主要来自丢弃模型重建的相位"
                        if share > 60 else
                        "推测**不成立**，幅度重分配才是主因，报告里的解释要改"))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_tracks": len(tracks),
        "tracks": [t.name for t in tracks],
        "means_usdr": means,
        "delta_phase_only": d_phase,
        "delta_magnitude_step": d_mag,
        "delta_total": total,
        "phase_share_pct": share,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
