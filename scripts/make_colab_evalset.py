"""从 MUSDB18-HQ 测试集切一个小评测包，供 Colab 训练中途评测用。

    python -m scripts.make_colab_evalset

完整测试集 10 GB，往云盘传不现实。但训练中途要的是**趋势**不是终值 ——
8 首 × 30 秒足够看出 SDR 在不在涨，体积能压到 100 MB 以内。

.. important::
   **这个包只用于训练中途看趋势，不能用来报最终数字。**
   正式评测必须在本机用 museval 跑完整 50 首（与项目其他数字同口径）。
   文件名里带 ``_probe`` 就是为了防止两者被混用。

.. warning::
   MUSDB18-HQ 是**非商业研究许可、不可再分发**。
   这个包传到你自己的云盘做研究用途没问题，**不要公开分享**。
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STEMS = ("mixture", "vocals", "drums", "bass", "other")


def main() -> int:
    p = argparse.ArgumentParser(description="生成 Colab 用的小评测包")
    p.add_argument("--src", default="data/musdb18hq/test")
    p.add_argument("--tracks", type=int, default=8)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--offset", type=float, default=30.0,
                   help="从每首第几秒开始截。默认跳过前 30 秒 —— "
                        "开头常是前奏或渐入，不能代表整首")
    p.add_argument("--out", default="data/musdb18hq_probe.tar")
    args = p.parse_args()

    import soundfile as sf

    src = ROOT / args.src
    if not src.exists():
        print(f"❌ 找不到 {src}")
        return 1

    dirs = sorted(d for d in src.iterdir() if d.is_dir())[: args.tracks]
    if not dirs:
        print(f"❌ {src} 下没有曲目目录")
        return 1

    stage = ROOT / "data" / "_probe_stage" / "test"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    n_files = 0
    for d in dirs:
        out_dir = stage / d.name
        out_dir.mkdir()
        for stem in STEMS:
            f = d / f"{stem}.wav"
            if not f.exists():
                print(f"  ⚠️ {d.name} 缺 {stem}.wav，跳过整首")
                shutil.rmtree(out_dir)
                break
            info = sf.info(str(f))
            start = int(min(args.offset, max(0, info.duration - args.seconds)) * info.samplerate)
            frames = int(args.seconds * info.samplerate)
            y, sr = sf.read(str(f), start=start, frames=frames, dtype="float32")
            # 存成 FLAC：无损，体积约为 WAV 的一半
            sf.write(str(out_dir / f"{stem}.wav"), y, sr, subtype="PCM_16")
            n_files += 1
        else:
            print(f"  ✅ {d.name}")

    out = ROOT / args.out
    with tarfile.open(out, "w") as tf:
        tf.add(stage, arcname="test")
    shutil.rmtree(stage.parent)

    mb = out.stat().st_size / 2**20
    print(f"\n✅ → {out}")
    print(f"   {len(dirs)} 首 × {args.seconds:.0f} 秒 = {n_files} 个文件，{mb:.0f} MB")
    print(f"\n下一步：把它传到云盘 Audio AI/MusicMixer/ 下，改名为 musdb18hq_test.tar")
    if mb > 500:
        print("⚠️ 超过 500 MB，上传会慢。减少 --tracks 或 --seconds。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
