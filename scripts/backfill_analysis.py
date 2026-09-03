"""给 ``web/mine/`` 里缺 analysis.json 的曲目补跑分析。

    python -m scripts.backfill_analysis

M6 之前分离的曲目没有分析结果（那时还没实现）。这个脚本只补分析，
**不重新分离** —— 分离要几十秒，分析只要几秒。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

WEB_MINE = Path(__file__).resolve().parent.parent / "web" / "mine"


def main() -> int:
    p = argparse.ArgumentParser(description="补跑分析")
    p.add_argument("--force", action="store_true", help="已有 analysis.json 也重跑")
    args = p.parse_args()

    import librosa

    from src.analysis.pipeline import ANALYSIS_SR, analyze_audio

    manifest = WEB_MINE / "manifest.json"
    cases = json.loads(manifest.read_text(encoding="utf-8"))["cases"] if manifest.exists() else []
    by_id = {c["id"]: c for c in cases}

    done = skipped = failed = 0
    for d in sorted(x for x in WEB_MINE.iterdir() if x.is_dir()):
        out = d / "analysis.json"
        if out.exists() and not args.force:
            skipped += 1
            continue
        mix = d / "mixture.mp3"
        if not mix.exists():
            print(f"  ⚠️ {d.name}: 没有 mixture.mp3，跳过")
            failed += 1
            continue
        try:
            y, sr = librosa.load(str(mix), sr=ANALYSIS_SR, mono=True)
            a = analyze_audio(y, sr, source=d.name)
            out.write_text(json.dumps(a, ensure_ascii=False, indent=2), encoding="utf-8")
            # manifest 里也要带上，否则前端不知道有分析结果
            if d.name in by_id:
                by_id[d.name].update({"analysis": f"mine/{d.name}/analysis.json",
                                      "bpm": a["bpm"], "key": a["key"]})
            print(f"  ✅ {d.name:<32}{a['bpm']:>6.1f} BPM  {a['key']:<10}"
                  f"{len(a['chords'])} 和弦  {len(a['segments'])} 段")
            done += 1
        except Exception as e:                          # noqa: BLE001
            print(f"  ❌ {d.name}: {type(e).__name__}: {e}")
            failed += 1

    if manifest.exists() and done:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["cases"] = [by_id.get(c["id"], c) for c in data["cases"]]
        manifest.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成 {done}，跳过 {skipped}，失败 {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
