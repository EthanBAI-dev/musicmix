"""多标签标签评测的单测。

重点验证 P4 赖以成立的那几条性质：
- mAP 与阈值无关，F1 强依赖阈值
- Macro 看长尾，Micro 被热门标签主导
- 退化标签在所有 macro 指标里被**一致地**排除
"""

import numpy as np
import pytest

from src.eval.tagging import (
    evaluate_tagging,
    hardest_tags,
    macro_average_precision,
    macro_f1,
    macro_roc_auc,
    micro_f1,
    per_tag_average_precision,
    per_tag_f1,
    tune_thresholds,
    valid_tag_mask,
)

RNG = np.random.default_rng(20260729)


def _separable_case():
    """构造一个刻意设计的两标签数据集，用来暴露"默认阈值 0.5 的危害"。

    - tag0（热门，50/100 正样本）：正样本得分 0.9、负样本 0.1 → 阈值 0.5 就完美
    - tag1（长尾，5/100 正样本）：正样本得分 0.45、负样本 0.05
      → **排序完美（AP=1.0），但没有任何一个样本能过 0.5**，F1@0.5 = 0

    所以：mAP = 1.0（两种阈值下都一样），Macro-F1@0.5 = 0.5，调完阈值 = 1.0。
    这就是 P4-L4 那张图要讲的全部故事。
    """
    n = 100
    y_true = np.zeros((n, 2))
    y_true[:50, 0] = 1
    y_true[:5, 1] = 1

    y_score = np.zeros((n, 2))
    y_score[:, 0] = np.where(y_true[:, 0] == 1, 0.9, 0.1)
    y_score[:, 1] = np.where(y_true[:, 1] == 1, 0.45, 0.05)
    return y_true, y_score


# --------------------------------------------------------------------------------------
# 排序类指标
# --------------------------------------------------------------------------------------

def test_perfect_ranking_gives_perfect_auc_and_map():
    y_true, y_score = _separable_case()
    assert macro_roc_auc(y_true, y_score) == pytest.approx(1.0)
    assert macro_average_precision(y_true, y_score) == pytest.approx(1.0)


def test_random_scores_give_auc_near_half():
    y_true = (RNG.random((2000, 10)) < 0.3).astype(float)
    y_score = RNG.random((2000, 10))
    assert macro_roc_auc(y_true, y_score) == pytest.approx(0.5, abs=0.05)


def test_inverted_scores_give_auc_near_zero():
    y_true = (RNG.random((500, 5)) < 0.4).astype(float)
    y_score = 1.0 - y_true + RNG.random((500, 5)) * 0.01
    assert macro_roc_auc(y_true, y_score) < 0.1


# --------------------------------------------------------------------------------------
# 阈值：P4 的核心洞察
# --------------------------------------------------------------------------------------

def test_threshold_tuning_lifts_macro_f1_but_not_map():
    """**本文件最重要的一条。**

    调阈值让 Macro-F1 从 0.5 变成 1.0，而 mAP 一字不变。
    """
    y_true, y_score = _separable_case()

    assert macro_f1(y_true, y_score, 0.5) == pytest.approx(0.5)

    th = tune_thresholds(y_true, y_score)
    assert macro_f1(y_true, y_score, th) == pytest.approx(1.0)

    # mAP 完全不受阈值影响 —— 它看的是排序
    assert macro_average_precision(y_true, y_score) == pytest.approx(1.0)


def test_tuned_threshold_lands_between_pos_and_neg_scores():
    y_true, y_score = _separable_case()
    th = tune_thresholds(y_true, y_score)
    assert 0.05 < th[1] <= 0.45, f"长尾标签的阈值应落在 (0.05, 0.45]，实际 {th[1]}"


def test_tune_thresholds_never_worse_than_default():
    """在**同一份数据**上，搜出来的阈值不可能比固定 0.5 差。

    （注意这只在训练/验证同源时成立；正式流程里阈值在验证集调、测试集用，
    所以测试集上偶尔略降是正常的，不是 bug。）
    """
    y_true = (RNG.random((300, 20)) < 0.15).astype(float)
    y_score = np.clip(y_true * 0.4 + RNG.random((300, 20)) * 0.5, 0, 1)
    th = tune_thresholds(y_true, y_score)
    assert macro_f1(y_true, y_score, th) >= macro_f1(y_true, y_score, 0.5) - 1e-9


def test_per_tag_f1_accepts_scalar_and_vector_thresholds():
    y_true, y_score = _separable_case()
    scalar = per_tag_f1(y_true, y_score, 0.5)
    vector = per_tag_f1(y_true, y_score, np.array([0.5, 0.5]))
    assert np.allclose(scalar, vector)


# --------------------------------------------------------------------------------------
# Macro vs Micro
# --------------------------------------------------------------------------------------

def test_micro_is_dominated_by_frequent_tag():
    """Micro-F1 被热门标签带高，Macro-F1 如实反映长尾被放弃了。

    这就是本项目主报 Macro 的理由。
    """
    y_true, y_score = _separable_case()
    assert micro_f1(y_true, y_score, 0.5) > macro_f1(y_true, y_score, 0.5)


# --------------------------------------------------------------------------------------
# 退化标签
# --------------------------------------------------------------------------------------

def test_valid_mask_excludes_all_zero_and_all_one_tags():
    y_true = np.zeros((10, 4))
    y_true[:, 0] = 1          # 全正 → 无效
    y_true[:5, 1] = 1         # 正常
    #  第 2 列全零 → 无效
    y_true[3, 3] = 1          # 正常
    mask = valid_tag_mask(y_true)
    assert mask.tolist() == [False, True, False, True]


def test_metrics_skip_degenerate_tags_consistently():
    """有效标签数要如实报出来，否则两次实验的 macro 值不可比。"""
    y_true = np.zeros((50, 3))
    y_true[:25, 0] = 1
    y_true[:10, 1] = 1
    # 第 2 列全零
    y_score = np.clip(y_true * 0.8 + 0.1, 0, 1)

    scores = evaluate_tagging(y_true, y_score)
    assert scores.n_tags == 3
    assert scores.n_valid_tags == 2
    assert np.isfinite(scores.map)


def test_per_tag_ap_returns_nan_for_degenerate():
    y_true = np.zeros((20, 2))
    y_true[:10, 0] = 1
    y_score = RNG.random((20, 2))
    ap = per_tag_average_precision(y_true, y_score)
    assert np.isfinite(ap[0])
    assert np.isnan(ap[1])


# --------------------------------------------------------------------------------------
# 汇总与分组：Stem-aware 假设检验的接口
# --------------------------------------------------------------------------------------

def test_evaluate_tagging_reports_both_threshold_variants():
    y_true, y_score = _separable_case()
    th = tune_thresholds(y_true, y_score)
    scores = evaluate_tagging(y_true, y_score, thresholds=th)

    assert scores.macro_f1_default == pytest.approx(0.5)
    assert scores.macro_f1_tuned == pytest.approx(1.0)
    assert scores.map == pytest.approx(1.0)


def test_tag_groups_give_separate_breakdowns():
    """P4-L5 就靠这个接口证明"收益主要来自乐器类标签"。"""
    y_true, y_score = _separable_case()
    scores = evaluate_tagging(
        y_true,
        y_score,
        tag_groups={"popular": np.array([0]), "long_tail": np.array([1])},
    )
    assert set(scores.groups) == {"popular", "long_tail"}
    assert scores.groups["popular"].macro_f1_default == pytest.approx(1.0)
    assert scores.groups["long_tail"].macro_f1_default == pytest.approx(0.0)


def test_evaluate_tagging_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        evaluate_tagging(np.zeros((10, 3)), np.zeros((10, 4)))


def test_hardest_tags_sorted_ascending_by_ap():
    """按 AP 升序返回最难的标签。

    Note:
        构造这个用例时踩了一次坑，值得记下来：最初把 mid 的正样本加 0.45、
        good 的加 0.9，以为 mid 会更差 —— 结果两者 AP 都是 1.0。
        因为**AP 只看排序，不看分数绝对值**，两种情况下正负样本都是完全分开的。
        要让 mid 真的更差，必须让它的正负样本分数**区间重叠**。
    """
    n = 200
    y_true = np.zeros((n, 3))
    y_true[:60, :] = 1

    y_score = RNG.random((n, 3)) * 0.1     # 全体基线落在 [0, 0.1]
    y_score[:60, 0] += 0.9                 # 完全分开 → AP = 1.0
    y_score[:60, 1] += 0.05                # 与负样本区间重叠 → AP 居中
    # 第 2 列不加任何提升 → 正负样本同分布 → AP ≈ 正样本率 0.3

    hard = hardest_tags(y_true, y_score, ["good", "mid", "bad"], k=3)
    assert [name for name, _, _ in hard] == ["bad", "mid", "good"]
    assert hard[0][2] == 60  # 正样本数如实带出
    assert hard[0][1] < hard[1][1] < hard[2][1]
