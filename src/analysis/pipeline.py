"""把节拍 / 调性 / 和弦 / 曲式串成一次完整分析。

产出的 dict 结构与 ``web/demo/analysis.json`` **完全一致**，
所以前端一行都不用改就能吃用户上传曲目的分析结果。

.. important::
   ``ground_truth`` 字段是给**合成演示曲**用的（那首曲子的 BPM/调性/和弦
   在合成时就是已知的）。真实曲目的分析是**估计值**，这个字段必须为 ``False``，
   前端会据此显示「模型估计」而不是「真值」。
   把估计值标成真值是最不该犯的错。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ANALYSIS_SR = 22050


def analyze_audio(y: np.ndarray, sr: int, source: str = "",
                  n_sections: int = 4) -> dict:
    """对单声道波形做完整分析。"""
    from src.analysis.beat import analyze_beats
    from src.analysis.chords import analyze_chords
    from src.analysis.key import analyze_key
    from src.analysis.structure import analyze_structure

    beat = analyze_beats(y, sr)
    key = analyze_key(y, sr)
    chords = analyze_chords(y, sr, beat.beats)
    sections = analyze_structure(y, sr, beat.beats, n_sections=n_sections)

    return {
        "source": source,
        "ground_truth": False,          # 估计值。见模块 docstring
        "duration": round(len(y) / sr, 3),
        "sample_rate": sr,
        **beat.as_dict(),
        **key.as_dict(),
        "chords": [c.as_dict() for c in chords],
        "segments": [s.as_dict() for s in sections],
    }


def analyze_file(path: str | Path, **kw) -> dict:
    import librosa

    p = Path(path)
    y, sr = librosa.load(str(p), sr=ANALYSIS_SR, mono=True)
    return analyze_audio(y, sr, source=p.stem, **kw)
