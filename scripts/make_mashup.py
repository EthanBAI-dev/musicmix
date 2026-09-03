"""自动 Mashup：取 A 的人声 + B 的伴奏，对齐速度、调性、小节线。

    python -m scripts.make_mashup --base 独自漂泊 --donor 某首歌
    python -m scripts.make_mashup --base 独自漂泊 --list      # 只看候选排名

曲目取自 ``web/mine/``（上传分离后的产物），需要已经有 stem 和 analysis.json。

**没有分离就没有 Mashup** —— 整首叠整首是噪音。这一步是 P1 分离能力的直接兑现。

.. note::
   本脚本不产出"好不好听"的分数。它报告的是**做了多少手脚**：
   变速百分比、变调半音数、小节线对齐误差。这些能算；好听与否不能。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.mashup.match import rank_candidates
from src.mashup.render import render_mashup

WEB_MINE = Path(__file__).resolve().parent.parent / "web" / "mine"
SR = 22050


def load_track(track_id: str) -> tuple[dict, Path]:
    d = WEB_MINE / track_id
    a = d / "analysis.json"
    if not a.exists():
        raise FileNotFoundError(
            f"{track_id} 没有 analysis.json。先通过网页上传分离一次 "
            f"（分离流程会自动跑分析）")
    return json.loads(a.read_text(encoding="utf-8")), d


def main() -> int:
    p = argparse.ArgumentParser(description="自动 Mashup")
    p.add_argument("--base", required=True, help="提供伴奏的曲目 id")
    p.add_argument("--donor", default="", help="提供人声的曲目 id。省略则用排名第一的")
    p.add_argument("--list", action="store_true", help="只列出候选排名，不渲染")
    p.add_argument("--max-stretch", type=float, default=0.08)
    p.add_argument("--max-semitones", type=int, default=2)
    p.add_argument("--donor-gain-db", type=float, default=-3.0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    import librosa
    import soundfile as sf

    base_a, base_dir = load_track(args.base)

    others = [d.name for d in WEB_MINE.iterdir()
              if d.is_dir() and d.name != args.base and (d / "analysis.json").exists()]
    if not others:
        print(f"❌ {WEB_MINE} 里没有别的带分析结果的曲目。至少要上传两首。")
        return 1

    donors = []
    for t in others:
        try:
            donors.append(load_track(t)[0])
        except FileNotFoundError:
            continue
    plans = rank_candidates(base_a, donors,
                            max_stretch=args.max_stretch, max_semitones=args.max_semitones)

    print(f"基底：{args.base}  {base_a['bpm']} BPM  {base_a['key']}\n")
    print(f"{'候选':<28}{'BPM':>7}{'变速':>9}{'变调':>7}  说明")
    for pl in plans:
        mark = "✅" if pl.feasible else "❌"
        note = pl.key_relation if pl.feasible else pl.reason
        print(f"  {mark} {pl.donor_id:<24}{pl.donor_bpm:>7.1f}"
              f"{pl.stretch_percent:>8.1f}%{pl.semitones:>+7.0f}  {note}")

    if args.list:
        return 0

    chosen = next((p for p in plans if p.donor_id == args.donor), None) if args.donor \
        else next((p for p in plans if p.feasible), None)
    if chosen is None:
        print("\n❌ 没有可行的候选。放宽 --max-stretch / --max-semitones 再试，"
              "但放得越宽音质损伤越大。")
        return 1
    if not chosen.feasible:
        print(f"\n⚠️ 指定的 {chosen.donor_id} 不可行：{chosen.reason}。仍然渲染，音质会明显受损。")

    _, donor_dir = load_track(chosen.donor_id)
    donor_a = json.loads((donor_dir / "analysis.json").read_text(encoding="utf-8"))

    # base 出伴奏（除人声外三轨），donor 出人声 —— 这正是分离的用处
    print(f"\n渲染：{args.base} 的伴奏 + {chosen.donor_id} 的人声")
    acc = None
    for stem in ("drums", "bass", "other"):
        f = base_dir / f"{stem}.mp3"
        if not f.exists():
            continue
        y, _ = librosa.load(str(f), sr=SR, mono=True)
        acc = y if acc is None else acc[: len(y)] + y[: len(acc)]
    voc_f = donor_dir / "vocals.mp3"
    if acc is None or not voc_f.exists():
        print("❌ 缺少 stem 文件（需要 base 的 drums/bass/other 与 donor 的 vocals）")
        return 1
    voc, _ = librosa.load(str(voc_f), sr=SR, mono=True)

    mixed, rep = render_mashup(acc, base_a, voc, donor_a, chosen, SR,
                               donor_gain_db=args.donor_gain_db)

    out = Path(args.out) if args.out else WEB_MINE / f"mashup_{args.base}_{chosen.donor_id}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), mixed, SR)

    print(f"\n实际做的手脚（这些能算；好不好听不能）：")
    print(f"  变速      {rep.stretch_applied:.4f}（{chosen.stretch_percent:+.1f}%）")
    print(f"  变调      {rep.semitones_applied:+.1f} 个半音")
    print(f"  对齐      {rep.n_bars_aligned} 个小节，误差中位 {rep.align_error_ms:.0f} ms")
    print(f"  人声增益  {rep.donor_gain_db:+.1f} dB")
    print(f"  时长      {rep.duration:.1f}s")
    print(f"\n✅ → {out}")
    print("\n对齐误差的下限由**节拍检测精度**决定，不是拉伸精度 —— "
          "两个相差 2% 的速度可能被检测成同一个值。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
