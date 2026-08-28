"""导出 P1 的失败案例 / 对照案例，供人工听审（wav + 网页用 mp3 + manifest）。

    python -m scripts.export_failure_cases                    # 默认最差 5 + 最好 2
    python -m scripts.export_failure_cases --worst 8 --best 3
    python -m scripts.export_failure_cases --no-separate      # 只重转 mp3，不重跑分离

产出：
- ``results/failure_cases/<case>/{stem}.wav`` 分离结果 + ``{stem}_TRUTH.wav`` 真值对照
- ``web/cases/<case>/*.mp3`` 网页用（256kbps）+ ``web/cases/manifest.json``

.. warning::
   **踩过的坑（2026-07-30）：截取窗口必须对三者一致。**

   第一版写成 ``seg = mix[a:b] if len(mix) > b else mix``，但真值那一行仍是
   ``truth[a:b]``。碰到比窗口终点还短的歌（``PR - Oh No`` 只有 76 秒）时，
   分离结果取了**整首 76 秒**、真值取了**最后 16 秒** —— A/B 听的压根不是同一段音乐，
   而且时长不一致这件事在播放器上并不显眼。

   现在窗口由 :func:`pick_window` 统一算一次，**mixture / 分离结果 / 真值共用**，
   并在导出后断言三者采样数完全相等。
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.audio.io import save_audio
from src.datasets.musdb import Track

SR = 44100
STEMS = ("vocals", "drums", "bass", "other")
OUT_WAV = Path("results/failure_cases")
OUT_WEB = Path("web/cases")
BITRATE = "256k"


def pick_window(n_samples: int, want_sec: float = 30.0, prefer_start_sec: float = 60.0) -> tuple[int, int]:
    """选一段长 ``want_sec`` 的窗口，尽量从 ``prefer_start_sec`` 开始。

    歌不够长时**整体前移**而不是截断，保证窗口长度稳定；
    连 ``want_sec`` 都不够的极短曲目才退化成整首。
    """
    want = int(want_sec * SR)
    if n_samples <= want:
        return 0, n_samples
    start = min(int(prefer_start_sec * SR), n_samples - want)
    return start, start + want


def to_mp3(src: Path, dst: Path) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-b:a", BITRATE, str(dst)],
        capture_output=True, check=False,
    )
    return r.returncode == 0


def export_case(track_name: str, tag: str, sep, want_sec: float) -> Path:
    """分离 + 切片 + 落盘。返回案例目录。"""
    from src.separation import demucs_model

    t = Track(name=track_name, path=Path("data/musdb18hq/test") / track_name, subset="test")
    mix = t.mixture()
    a, b = pick_window(mix.shape[0], want_sec)
    seg = mix[a:b]

    stems = demucs_model.separate(sep, seg)

    case_dir = OUT_WAV / f"{tag}_{track_name.replace('/', '_')[:44]}"
    save_audio(case_dir / "mixture.wav", seg)
    for name, y in stems.items():
        save_audio(case_dir / f"{name}.wav", y)
    for name in STEMS:
        save_audio(case_dir / f"{name}_TRUTH.wav", t.audio(name)[a:b])

    # 断言窗口一致 —— 这正是上一版漏掉的检查
    n = seg.shape[0]
    for f in case_dir.glob("*.wav"):
        import soundfile as sf
        frames = sf.info(str(f)).frames
        assert frames == n, f"{f.name} 有 {frames} 采样，与窗口 {n} 不一致"
    return case_dir


def build_manifest(case_dirs: list[tuple[Path, str, str]]) -> None:
    df = pd.read_csv("results/p1_htdemucs.csv").set_index("track")
    energy = pd.read_csv("results/p1_stem_energy.csv").set_index("track")
    med = {s: float(energy[f"{s}_rel_dB"].median()) for s in STEMS}

    cases = []
    for case_dir, tag, track in case_dirs:
        row, en = df.loc[track], energy.loc[track]
        web_dir = OUT_WEB / case_dir.name

        files = {}
        for stem in (*STEMS, "mixture"):
            if to_mp3(case_dir / f"{stem}.wav", web_dir / f"{stem}.mp3"):
                files[stem] = f"cases/{case_dir.name}/{stem}.mp3"
            truth = case_dir / f"{stem}_TRUTH.wav"
            if truth.exists() and to_mp3(truth, web_dir / f"{stem}_truth.mp3"):
                files[f"{stem}_truth"] = f"cases/{case_dir.name}/{stem}_truth.mp3"

        # 逐声部标注：低 SDR 是「真实算法失败」还是「该轨本来就很轻」造成的指标假象
        diag = {}
        for stem in STEMS:
            rel = float(en[f"{stem}_rel_dB"])
            diag[stem] = {
                "csdr": round(float(row[f"cSDR_{stem}"]), 2),
                "energy_rel_db": round(rel, 1),
                "energy_median_db": round(med[stem], 1),
                "quiet": bool(rel < med[stem] - 6),
            }

        cases.append({
            "id": case_dir.name, "track": track, "tag": tag,
            "csdr_mean": round(float(row["cSDR_mean"]), 2),
            "stems": diag, "files": files,
        })
        print(f"  ✅ {tag:5s} {row['cSDR_mean']:5.2f} dB  {track}")

    cases.sort(key=lambda c: c["csdr_mean"])
    OUT_WEB.mkdir(parents=True, exist_ok=True)
    (OUT_WEB / "manifest.json").write_text(
        json.dumps({
            "note": "P1 htdemucs 在 MUSDB18-HQ test 上的失败/对照案例。"
                    "mixture / 分离结果 / 真值取自**同一个时间窗口**（见 pick_window）。"
                    "mp3 256kbps —— 分离伪影远大于编码伪影，不影响听辨；"
                    "严格听审请用 results/failure_cases/ 下的原始 wav。",
            "cases": cases,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    size = sum(f.stat().st_size for f in OUT_WEB.rglob("*.mp3")) / 1e6
    print(f"\n✅ {len(cases)} 个案例 → {OUT_WEB}（{size:.0f} MB）")


def main() -> int:
    p = argparse.ArgumentParser(description="导出失败案例供听审")
    p.add_argument("--worst", type=int, default=5)
    p.add_argument("--best", type=int, default=2)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--no-separate", action="store_true", help="复用已有 wav，只重转 mp3 与 manifest")
    args = p.parse_args()

    df = pd.read_csv("results/p1_htdemucs.csv").sort_values("cSDR_mean")
    picks = ([(n, "worst") for n in df.track.head(args.worst)]
             + [(n, "best") for n in df.track.tail(args.best)])

    case_dirs = []
    if args.no_separate:
        for name, tag in picks:
            d = OUT_WAV / f"{tag}_{name.replace('/', '_')[:44]}"
            if d.is_dir():
                case_dirs.append((d, tag, name))
            else:
                print(f"⚠️  跳过 {name}：{d} 不存在（去掉 --no-separate 重新分离）")
    else:
        from src.separation import demucs_model
        sep = demucs_model.load("htdemucs")
        print(f"已加载 htdemucs → {sep.device}\n")
        for name, tag in picks:
            case_dirs.append((export_case(name, tag, sep, args.seconds), tag, name))

    build_manifest(case_dirs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
