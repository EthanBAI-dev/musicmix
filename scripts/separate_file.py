"""把任意一首歌分离成四轨，并直接放进混音台可以加载的位置。

    python -m scripts.separate_file ~/Music/某首歌.mp3
    python -m scripts.separate_file 某首歌.mp3 --seconds 60      # 只处理前 60 秒
    python -m scripts.separate_file 某首歌.mp3 --model htdemucs_ft

这是把 P1 的分离能力接到 P7 混音台的最后一段路。在 P6（FastAPI 上传接口）
做出来之前，它就是"导入一首歌 → 拆四轨 → 在网页里混音"的完整流程。

产出 ``web/mine/<曲名>/{mixture,vocals,drums,bass,other}.mp3`` 与
``web/mine/manifest.json``；网页会自动列出来。

.. note::
   **全曲分离的耗时按 RTF 0.066 算**（M2 Max / MPS 实测）：
   一首 4 分钟的歌约 16 秒。``--seconds`` 可以只处理开头一段，
   用来快速试听效果。
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

import numpy as np

OUT_DIR = Path("web/mine")
STEMS = ("vocals", "drums", "bass", "other")


def slugify(name: str) -> str:
    """文件名 → 安全的目录名。保留中文，只去掉路径分隔符和控制字符。"""
    s = re.sub(r"[/\\:*?\"<>|\x00-\x1f]", "_", name).strip(" .")
    return s[:60] or "untitled"


def to_mp3(src: Path, dst: Path, bitrate: str = "256k") -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-b:a", bitrate, str(dst)],
        capture_output=True, check=False,
    )
    return r.returncode == 0


def update_manifest(entry: dict) -> None:
    """把新曲目并进 manifest；同名覆盖，顺序按加入时间倒序（新的在前）。"""
    mf = OUT_DIR / "manifest.json"
    data = {"note": "本地导入并分离的曲目（scripts/separate_file.py）", "cases": []}
    if mf.exists():
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            pass          # manifest 坏了不该阻断分离，重建即可
    cases = [c for c in data.get("cases", []) if c["id"] != entry["id"]]
    data["cases"] = [entry] + cases
    mf.parent.mkdir(parents=True, exist_ok=True)
    mf.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="分离一首歌并放进混音台")
    p.add_argument("audio", help="任意音频文件（mp3 / wav / flac / m4a …）")
    p.add_argument("--model", default="htdemucs",
                   help="htdemucs（默认，快）/ htdemucs_ft（官方 9.20 dB，慢 4 倍）")
    p.add_argument("--seconds", type=float, default=0,
                   help="只处理开头 N 秒；0 表示整首")
    p.add_argument("--device", default="auto")
    p.add_argument("--keep-wav", action="store_true", help="同时保留无损 wav")
    args = p.parse_args()

    src = Path(args.audio).expanduser()
    if not src.exists():
        print(f"❌ 找不到文件：{src}")
        return 1

    from src.audio.io import load_audio, save_audio
    from src.separation import demucs_model

    print(f"读取 {src.name}")
    mix, sr = load_audio(src, stereo=True, normalize_loudness=False)
    total = mix.shape[0] / sr
    if args.seconds and args.seconds < total:
        mix = mix[: int(args.seconds * sr)]
    dur = mix.shape[0] / sr
    print(f"  时长 {total:.0f}s" + (f"，处理前 {dur:.0f}s" if dur < total else "") + f"，{sr} Hz 立体声")

    sep = demucs_model.load(args.model, device=args.device)
    print(f"  分离模型 {sep.name}（{sep.device}）…")
    t0 = time.perf_counter()
    stems = demucs_model.separate(sep, mix)
    el = time.perf_counter() - t0
    print(f"  完成，耗时 {el:.1f}s（RTF {el/dur:.3f}，×{dur/el:.1f} 实时）")

    # Demucs 不是掩码方法，四轨之和不会精确等于混音；这个数只用来抓离谱错误
    resid = float(np.max(np.abs(mix[: stems["vocals"].shape[0]] - sum(stems.values()))))
    print(f"  自检 |混音 − Σ四轨| 最大 {resid:.4f}（Demucs 非掩码方法，不应为 0）")

    slug = slugify(src.stem)
    web_dir = OUT_DIR / slug
    tmp = web_dir / "_tmp.wav"

    files = {}
    for name, y in [("mixture", mix), *stems.items()]:
        save_audio(tmp, y, sr, subtype="PCM_16")
        if to_mp3(tmp, web_dir / f"{name}.mp3"):
            files[name] = f"mine/{slug}/{name}.mp3"
        if args.keep_wav:
            save_audio(web_dir / f"{name}.wav", y, sr)
    tmp.unlink(missing_ok=True)

    rms = {k: float(np.sqrt(np.mean(v**2))) for k, v in stems.items()}
    update_manifest({
        "id": slug,
        "track": src.stem,
        "tag": "mine",
        "csdr_mean": None,                 # 没有真值，算不了 SDR
        "duration": round(dur, 1),
        "model": sep.name,
        # 各轨能量：没有真值时，这是唯一能提供的客观信息
        "stem_rms_db": {k: round(20 * np.log10(v + 1e-12), 1) for k, v in rms.items()},
        "files": files,
    })

    size = sum(f.stat().st_size for f in web_dir.glob("*.mp3")) / 2**20
    print(f"\n✅ {web_dir}（{size:.0f} MB）")
    for k in STEMS:
        print(f"   {k:<8}{20*np.log10(rms[k]+1e-12):6.1f} dBFS")
    print("\n打开混音台：python -m scripts.serve  →  http://localhost:8123/web/index.html")
    print("在「我的曲目」下拉里选它。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
