"""音乐分析（节拍/调性/和弦/曲式）的单测。

**用合成信号做夹具，真值由构造方式决定** —— 这是 P0 定下的做法：
在玩具数据上，正确答案是**算出来的**而不是标注的，
所以测试能断言「恰好等于」而不是「看起来差不多」。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.beat import (
    BeatResult,
    estimate_downbeats,
    evaluate_beats,
    extend_beats,
)
from src.analysis.chords import Chord, evaluate_chords, to_mir_eval_label
from src.analysis.key import KK_MAJOR, KK_MINOR, PITCH_NAMES, analyze_key
from src.analysis.structure import analyze_structure

SR = 22050


# ---------------- 拍点外推 ----------------

def test_extend_fills_missing_first_beat():
    """节拍跟踪常常漏掉开头第一拍 —— 外推必须把它补回来。

    这条测试对应一个实测到的真实问题：漏掉首拍会让**后续所有和弦区间
    整体偏移一拍**，实测把和弦加权准确率从 0.924 拉低到 0.688。
    """
    period = 0.5
    detected = np.arange(1, 10) * period          # 从 0.5s 开始，漏了 0.0
    out = extend_beats(detected, duration=5.0)
    assert out[0] == pytest.approx(0.0, abs=1e-9), "首拍必须被补到 0"
    assert len(out) > len(detected)


def test_extend_does_not_move_detected_beats():
    """外推只能**添加**，不能改动已检测到的拍。"""
    detected = np.array([1.0, 1.5, 2.0, 2.5])
    out = extend_beats(detected, duration=4.0)
    for t in detected:
        assert np.min(np.abs(out - t)) < 1e-9


def test_extend_covers_tail():
    detected = np.array([0.0, 0.5, 1.0])
    out = extend_beats(detected, duration=3.0)
    assert out[-1] > 2.4, "尾部也要补到接近曲末"


def test_extend_ignores_too_few_beats():
    assert len(extend_beats(np.array([1.0]), 10.0)) == 1


# ---------------- 小节线 ----------------

def test_downbeat_finds_accented_phase():
    """构造一个「每 4 拍第 2 拍最重」的起始强度曲线，相位必须被找出来。"""
    n_beats, bpb, phase = 16, 4, 1
    beats = np.arange(n_beats) * 0.5
    env = np.zeros(2000)
    for i, t in enumerate(beats):
        f = int(round(t * SR / 512))
        env[f] = 5.0 if i % bpb == phase else 1.0

    down, conf = estimate_downbeats(env, beats, SR, bpb)
    assert down[0] == pytest.approx(beats[phase])
    assert conf > 1.5, "重音如此明显，置信度不该接近 1"


def test_downbeat_confidence_is_one_when_flat():
    """所有拍一样重时，**没有任何依据**判断相位 —— 置信度必须报 1.0。

    这比随便挑一个相位再声称有把握重要得多：
    界面可以据此显示「不确定」，而不是假装有结论。
    """
    beats = np.arange(16) * 0.5
    env = np.ones(2000)
    _, conf = estimate_downbeats(env, beats, SR, 4)
    assert conf == pytest.approx(1.0, abs=1e-6)


def test_downbeat_empty_input():
    down, conf = estimate_downbeats(np.ones(100), np.array([]), SR, 4)
    assert len(down) == 0 and conf == 0.0


# ---------------- 调性 ----------------

def _tone(freqs, seconds=4.0, sr=SR):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs)
    return (y / np.max(np.abs(y))).astype(np.float32)


def test_key_detects_c_major_triad():
    # C4 E4 G4 —— 最没有歧义的 C 大调材料
    r = analyze_key(_tone([261.63, 329.63, 392.00]), SR)
    assert r.key in ("C major", "A minor"), f"得到 {r.key}"


def test_key_reports_relative_alternative():
    """相对大小调共用音级集合，方法上无法区分 —— 必须把另一个可能一并报出。"""
    r = analyze_key(_tone([261.63, 329.63, 392.00]), SR)
    assert r.relative_alternative != r.key
    # 相对调的主音必须相差 3 个半音
    a = PITCH_NAMES.index(r.key.split()[0])
    b = PITCH_NAMES.index(r.relative_alternative.split()[0])
    assert (a - b) % 12 in (3, 9)


def test_kk_profiles_are_the_published_ones():
    """轮廓是 Krumhansl-Kessler 1982 的实验数据，不该被随手改动。"""
    assert KK_MAJOR[0] == pytest.approx(6.35)
    assert KK_MINOR[0] == pytest.approx(6.33)
    assert len(KK_MAJOR) == len(KK_MINOR) == 12


# ---------------- 和弦标签转换 ----------------

@pytest.mark.parametrize("shown,mir", [
    ("Am", "A:min"), ("F", "F:maj"), ("C#m", "C#:min"), ("G", "G:maj"), ("N", "N"),
])
def test_chord_label_conversion(shown, mir):
    """界面用乐手记法、评测用 mir_eval 语法，两套刻意分开。

    这条测试来自一个真实报错：把 "Am" 直接喂给 mir_eval 会抛
    InvalidChordException。
    """
    assert to_mir_eval_label(shown) == mir


def test_chord_eval_perfect_match():
    ref = [{"start": 0.0, "end": 2.0, "label": "Am"},
           {"start": 2.0, "end": 4.0, "label": "F"}]
    est = [Chord(0.0, 2.0, "Am", 1.0), Chord(2.0, 4.0, "F", 1.0)]
    assert evaluate_chords(ref, est)["weighted_accuracy"] == pytest.approx(1.0)


def test_chord_eval_all_wrong():
    ref = [{"start": 0.0, "end": 4.0, "label": "Am"}]
    est = [Chord(0.0, 4.0, "F", 1.0)]
    assert evaluate_chords(ref, est)["weighted_accuracy"] == pytest.approx(0.0)


def test_chord_eval_half_shifted():
    """区间错位一半时，准确率应当明显掉下来 —— 这正是漏掉首拍的后果。"""
    ref = [{"start": 0.0, "end": 2.0, "label": "Am"},
           {"start": 2.0, "end": 4.0, "label": "F"}]
    est = [Chord(0.0, 3.0, "Am", 1.0), Chord(3.0, 4.0, "F", 1.0)]
    acc = evaluate_chords(ref, est)["weighted_accuracy"]
    assert 0.4 < acc < 0.9, f"错位应当被扣分但不至于归零，得到 {acc}"


# ---------------- 拍点评测 ----------------

def test_beat_eval_perfect():
    beats = np.arange(0, 30, 0.5)
    assert evaluate_beats(beats, beats)["f_measure"] == pytest.approx(1.0)


def test_beat_eval_double_tempo_hurts_cmlt_not_amlt():
    """倍频错误：CMLt 应当掉，AMLt 应当仍高 —— 两者的差就是倍频错误的量。"""
    ref = np.arange(0, 30, 0.5)
    est = np.arange(0, 30, 0.25)           # 两倍速
    s = evaluate_beats(ref, est)
    assert s["amlt"] > s["cmlt"], "AMLt 容忍倍频，CMLt 不容忍"


def test_beat_eval_empty():
    assert evaluate_beats(np.array([]), np.arange(10) * 0.5)["f_measure"] == 0.0


# ---------------- 曲式 ----------------

def test_structure_short_audio_returns_single_section():
    """太短就不该硬切 —— 切出一堆碎片不如老实说「只有一段」。"""
    y = _tone([440.0], seconds=3.0)
    out = analyze_structure(y, SR, np.arange(0, 3, 0.5))
    assert len(out) == 1 and out[0].label == "A"


def test_structure_does_not_invent_verse_chorus():
    """段落标签只能是 intro/outro 或字母 —— **不许**出现 verse/chorus。

    算法能判断「这两段像」，但无法知道那叫副歌。
    编造一个更专业的名字会让界面显得聪明，代价是给出假信息。
    """
    y = _tone([440.0], seconds=3.0)
    for s in analyze_structure(y, SR, np.arange(0, 3, 0.5)):
        assert s.label in ("intro", "outro") or (len(s.label) == 1 and s.label.isalpha())


def test_beat_result_as_dict_shape():
    r = BeatResult(tempo=120.0, beats=np.arange(8) * 0.5,
                   downbeats=np.arange(2) * 2.0, beats_per_bar=4)
    d = r.as_dict()
    assert d["bpm"] == 120.0 and d["time_signature"] == "4/4"
    assert len(d["beats"]) == 8 and r.n_bars == 2
