"""Mashup 配对：判断两首歌能不能拼在一起，以及代价有多大。

拼歌的三个硬约束：

1. **速度**要对齐 —— 拉伸幅度越大，相位声码器的瑕疵越明显
2. **调性**要相容 —— 不相容的话人声和伴奏会打架
3. **小节线**要对齐 —— 差半拍就完全不成立

本模块只做**打分与筛选**，不做音频处理（那在 :mod:`src.mashup.render`）。

.. note::
   **Mashup 没有客观真值。** 没有"这两首拼起来好不好听"的标注数据，
   所以这里不产出任何"质量分"。产出的全是**可测量的代价**：
   拉伸了百分之多少、变调了几个半音、小节线对齐误差多少毫秒。
   这些数字不告诉你好不好听，但它们**如实描述了做了多少手脚** ——
   一个编出来的"兼容度 87 分"只会让人以为有依据。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.analysis.key import PITCH_NAMES

# 拉伸超过这个比例，相位声码器的金属感就藏不住了。
# 8% 是 DJ 器材上常见的 pitch fader 范围（±8%），沿用这个约定。
MAX_STRETCH = 0.08

# 五度圈上的位置：C G D A E B F# C# G# D# A# F
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]


def circle_distance(tonic_a: int, tonic_b: int) -> int:
    """两个主音在五度圈上的最短距离（0~6）。

    五度圈相邻的调共享 6 个音，听感上最接近；相距越远共享音越少。
    """
    ia, ib = CIRCLE_OF_FIFTHS.index(tonic_a), CIRCLE_OF_FIFTHS.index(tonic_b)
    d = abs(ia - ib) % 12
    return min(d, 12 - d)


def key_distance(key_a: str, key_b: str) -> tuple[int, str]:
    """调性距离，以及**为什么**。

    Returns:
        ``(需要变调的半音数, 关系说明)``。半音数取绝对值最小的那个方向。
    """
    ta, ma = PITCH_NAMES.index(key_a.split()[0]), key_a.split()[1]
    tb, mb = PITCH_NAMES.index(key_b.split()[0]), key_b.split()[1]

    semitones = (tb - ta + 6) % 12 - 6           # 落在 [-6, 5]
    if ta == tb and ma == mb:
        return 0, "同调"
    if ma != mb:
        # 相对大小调：小调主音 = 大调主音 + 9（如 C大 ↔ a小）
        rel = (ta + 9) % 12 if ma == "major" else (ta + 3) % 12
        if rel == tb:
            return 0, "相对大小调（同音级集合）"
        return semitones, f"异调式，五度圈距离 {circle_distance(ta, tb)}"
    return semitones, f"同调式，五度圈距离 {circle_distance(ta, tb)}"


@dataclass
class MashupPlan:
    """把 ``donor`` 拼到 ``base`` 上需要做的手脚。"""

    base_id: str
    donor_id: str
    base_bpm: float
    donor_bpm: float
    # **直接就是 librosa.effects.time_stretch 的 rate 参数**，不做二次换算。
    # 曾经把它定义成 donor_bpm/base_bpm（方向相反），显示的百分比是对的、
    # 传进去的值是反的 —— 靠"对齐误差恰好等于一拍"才发现。
    # 一个量只该有一种含义，中间隔一次取倒数就迟早会错。
    stretch: float               # <1 = 放慢 donor
    semitones: float             # donor 要变调多少半音
    key_relation: str
    feasible: bool
    reason: str = ""

    @property
    def stretch_percent(self) -> float:
        """donor 速度变化百分比。负 = 放慢。"""
        return (self.stretch - 1.0) * 100

    def as_dict(self) -> dict:
        return {"base": self.base_id, "donor": self.donor_id,
                "base_bpm": round(self.base_bpm, 1), "donor_bpm": round(self.donor_bpm, 1),
                "stretch": round(self.stretch, 4),
                "speed_change_percent": round(self.stretch_percent, 2),
                "semitones": round(self.semitones, 2),
                "key_relation": self.key_relation,
                "feasible": self.feasible, "reason": self.reason}


def _bar_seconds(analysis: dict) -> float:
    """从**实测小节线**算平均小节时长。拿不到就返回 0。

    比用 BPM 标量更可靠：BPM 是一个四舍五入过的估计，
    0.1% 的误差在 233 秒的曲子上会累积成 233 ms 的漂移；
    而小节线是逐个测出来的，中位间隔天然吸收了这个误差。
    """
    import numpy as np

    d = np.asarray(analysis.get("downbeats") or [], dtype=float)
    return float(np.median(np.diff(d))) if len(d) >= 3 else 0.0


def plan_mashup(base: dict, donor: dict, max_stretch: float = MAX_STRETCH,
                max_semitones: int = 2, use_downbeats: bool = True) -> MashupPlan:
    """给出把 donor 拼到 base 上的方案，并判断是否可行。

    Args:
        base, donor: :func:`src.analysis.pipeline.analyze_audio` 的输出
        max_stretch: 允许的最大变速比例
        max_semitones: 允许的最大变调半音数。超过 2 个半音，
            人声的共振峰会明显失真（librosa 的 pitch_shift 不做共振峰保持）
    """
    bb, db = float(base["bpm"]), float(donor["bpm"])
    if bb <= 0 or db <= 0:
        return MashupPlan(base.get("source", "?"), donor.get("source", "?"), bb, db,
                          1.0, 0.0, "未知", False, "速度未知（分析失败？）")

    # time_stretch(rate=r) 之后 donor 的速度变成 db*r，要它等于 bb → r = bb/db
    ratio = bb / db
    if use_downbeats:
        # 有实测小节线时优先用它：donor 一个小节的时长要拉成 base 的
        bs, ds = _bar_seconds(base), _bar_seconds(donor)
        if bs > 0 and ds > 0:
            ratio = ds / bs
    # 倍频修正：117 vs 60 BPM 其实是同一个速度层级，不该判为不兼容
    for mult in (0.5, 1.0, 2.0):
        if abs(ratio * mult - 1.0) < abs(ratio - 1.0):
            ratio = ratio * mult
    stretch = ratio

    semitones, relation = key_distance(base["key"], donor["key"])

    # **五度圈距离小 ≠ 变调半音数小**：C 大调与 G 大调在五度圈上相邻，
    # 但对齐主音要移 5 个半音。而 DJ 的谐波混音实践里，相邻调**根本不用变调**
    # 就能叠 —— 它们共享 6 个音，冲突的只有一个。
    # 所以这里比较两条路径，取代价小的那条：
    #   路径A 变调到同一个调；路径B 不变调，靠调性相容
    ta = PITCH_NAMES.index(base["key"].split()[0])
    tb = PITCH_NAMES.index(donor["key"].split()[0])
    same_mode = base["key"].split()[1] == donor["key"].split()[1]
    if semitones != 0 and same_mode and circle_distance(ta, tb) <= 1:
        semitones = 0.0
        relation += "（相邻调，不变调直接叠）"

    over_stretch = abs(stretch - 1.0) > max_stretch
    over_pitch = abs(semitones) > max_semitones
    reasons = []
    if over_stretch:
        reasons.append(f"变速 {abs(stretch-1)*100:.1f}% 超过上限 {max_stretch*100:.0f}%")
    if over_pitch:
        reasons.append(f"变调 {abs(semitones):.0f} 个半音超过上限 {max_semitones}")

    return MashupPlan(base.get("source", "?"), donor.get("source", "?"), bb, db,
                      stretch, float(semitones), relation,
                      feasible=not reasons, reason="；".join(reasons))


def rank_candidates(base: dict, donors: list[dict], **kw) -> list[MashupPlan]:
    """把候选按「手脚做得最少」排序 —— 可行的在前。

    排序键刻意是**代价**而不是某种"匹配度"：代价是能算的，匹配度不是。
    """
    plans = [plan_mashup(base, d, **kw) for d in donors]
    return sorted(plans, key=lambda p: (not p.feasible,
                                        abs(p.semitones),
                                        abs(p.stretch - 1.0)))
