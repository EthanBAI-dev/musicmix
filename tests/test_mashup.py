"""Mashup 配对与渲染的单测。

**变速方向**是这里最容易错、也最不容易发现的地方：
方向反了的话显示的百分比仍然是对的，只有对齐误差会暴露它
（实测那次误差恰好等于一拍）。所以这里直接断言方向。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mashup.match import circle_distance, key_distance, plan_mashup, rank_candidates
from src.mashup.render import align_to_downbeat, render_mashup, time_stretch

SR = 22050


def A(bpm, key, src="x", n_bars=4, beats_per_bar=4):
    """构造一份分析结果。

    **小节线必须由 bpm 算出来**，不能写死。
    第一版夹具给所有曲目都写 ``[0, 2, 4]``（那是 120 BPM 的小节），
    而 bpm 字段写着 100/110 —— 数据自相矛盾。
    plan_mashup 优先采信实测小节线，于是比率恒为 1.0，五条测试一起挂。
    夹具自洽比测试通过重要。
    """
    bar = beats_per_bar * 60.0 / bpm if bpm > 0 else 2.0
    return {"bpm": bpm, "key": key, "source": src,
            "downbeats": [round(i * bar, 6) for i in range(n_bars)]}


# ---------------- 调性 ----------------

def test_same_key_needs_no_shift():
    assert key_distance("C major", "C major")[0] == 0


def test_relative_keys_need_no_shift():
    """相对大小调共用同一组音 —— 不该变调。"""
    assert key_distance("C major", "A minor")[0] == 0
    assert key_distance("A minor", "C major")[0] == 0


def test_circle_distance_is_symmetric_and_bounded():
    for a in range(12):
        for b in range(12):
            d = circle_distance(a, b)
            assert 0 <= d <= 6
            assert d == circle_distance(b, a)


def test_adjacent_keys_play_without_shift():
    """五度圈相邻的调共享 6 个音，DJ 实践里直接叠 —— 不该被判成要移 5 个半音。"""
    p = plan_mashup(A(120, "C major"), A(120, "G major"))
    assert p.semitones == 0 and p.feasible


def test_distant_key_is_rejected():
    p = plan_mashup(A(120, "C major"), A(120, "F# major"))
    assert not p.feasible


# ---------------- 变速方向 ----------------

def test_faster_donor_is_slowed_down():
    """donor 比 base 快 → rate 必须 < 1（放慢）。

    这条对应一个真实 bug：曾经把 stretch 定义成 donor/base（方向相反），
    显示的百分比是对的、传给 librosa 的值是反的。
    """
    p = plan_mashup(A(100, "C major"), A(110, "C major"))
    assert p.stretch < 1.0, f"donor 更快就该放慢，得到 rate={p.stretch}"
    assert p.stretch_percent < 0


def test_slower_donor_is_sped_up():
    p = plan_mashup(A(110, "C major"), A(100, "C major"))
    assert p.stretch > 1.0 and p.stretch_percent > 0


def test_stretch_actually_matches_tempo():
    """rate 用下去之后，donor 的速度必须真的等于 base。"""
    base_bpm, donor_bpm = 96.0, 100.0
    p = plan_mashup(A(base_bpm, "C major"), A(donor_bpm, "C major"))
    assert donor_bpm * p.stretch == pytest.approx(base_bpm, rel=1e-6)


def test_octave_error_is_corrected():
    """192 与 96 是同一个速度层级，不该被判成变速 100%。"""
    p = plan_mashup(A(96, "C major"), A(192, "C major"))
    assert p.stretch == pytest.approx(1.0) and p.feasible


def test_large_tempo_gap_is_rejected():
    p = plan_mashup(A(96, "C major"), A(120, "C major"))
    assert not p.feasible and "变速" in p.reason


def test_zero_bpm_is_rejected_not_crashed():
    """分析失败时 bpm 会是 0 —— 必须判为不可行，而不是抛除零异常。"""
    p = plan_mashup(A(96, "C major"), A(0.0, "C major"))
    assert not p.feasible


# ---------------- 渲染 ----------------

def test_time_stretch_changes_duration_in_right_direction():
    y = np.zeros(SR * 4, dtype=np.float32)
    assert len(time_stretch(y, 0.5, SR)) > len(y)      # 放慢 → 变长
    assert len(time_stretch(y, 2.0, SR)) < len(y)


def test_align_shifts_forward_and_back():
    y = np.arange(10, dtype=np.float32)
    assert len(align_to_downbeat(y, 1, 0.0, 3.0)) == 13
    assert len(align_to_downbeat(y, 1, 3.0, 0.0)) == 7
    assert (align_to_downbeat(y, 1, 1.0, 1.0) == y).all()


def test_render_output_never_clips():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.5, SR * 4).astype(np.float32)
    donor = rng.normal(0, 0.5, SR * 4).astype(np.float32)
    p = plan_mashup(A(100, "C major"), A(100, "C major"))
    mixed, _ = render_mashup(base, A(100, "C major"), donor, A(100, "C major"), p, SR)
    assert np.max(np.abs(mixed)) <= 1.0 + 1e-6


def test_render_reports_alignment_error():
    base = np.zeros(SR * 4, dtype=np.float32)
    donor = np.zeros(SR * 4, dtype=np.float32)
    p = plan_mashup(A(100, "C major"), A(100, "C major"))
    _, rep = render_mashup(base, A(100, "C major"), donor, A(100, "C major"), p, SR)
    assert rep.n_bars_aligned == 3
    assert rep.align_error_ms == pytest.approx(0.0, abs=1.0)


def test_fixture_downbeats_agree_with_bpm():
    """夹具自检：小节线间隔必须与 bpm 一致，否则后面的测试测的是矛盾数据。"""
    a = A(120, "C major")
    gaps = np.diff(a["downbeats"])
    assert gaps == pytest.approx(4 * 60.0 / 120)


def test_plan_prefers_measured_downbeats_over_bpm_scalar():
    """bpm 与小节线冲突时，应当采信**实测的小节线** ——
    bpm 是四舍五入过的标量，小节线是逐个测出来的。"""
    base = A(100, "C major")
    donor = A(100, "C major")
    donor["downbeats"] = [round(i * 2.4, 6) for i in range(4)]   # 实际更慢
    p = plan_mashup(base, donor)
    assert p.stretch == pytest.approx(2.4 / (4 * 60.0 / 100))


def test_ranking_puts_feasible_first():
    base = A(100, "C major")
    donors = [A(140, "C major", "太快"), A(101, "C major", "刚好"),
              A(100, "F# major", "调不合")]
    ranked = rank_candidates(base, donors)
    assert ranked[0].donor_id == "刚好" and ranked[0].feasible
    assert not ranked[-1].feasible
