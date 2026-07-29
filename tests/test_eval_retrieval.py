"""检索评测的单测。

重点验证：
- HitRate 与 true Recall 是**两个不同的东西**（这是本项目要求分开命名的理由）
- 检索时"查询自己"确实被剔除了（忘掉这一步会让 HitRate@1 恒等于 1）
- NDCG 的理想增益来自全库，不是来自检索结果自身
"""

import numpy as np
import pytest

from src.eval.retrieval import (
    average_precision_at_k,
    build_relevance,
    evaluate_retrieval,
    hit_rate_at_k,
    ideal_gains,
    index_recall,
    jaccard_at_k,
    l2_normalize,
    ndcg_at_k,
    precision_at_k,
    rank_by_similarity,
    recall_at_k,
    shared_tag_counts,
)

RNG = np.random.default_rng(20260729)


# --------------------------------------------------------------------------------------
# 排序构造
# --------------------------------------------------------------------------------------

def test_l2_normalize_gives_unit_vectors():
    x = RNG.standard_normal((7, 16))
    assert np.allclose(np.linalg.norm(l2_normalize(x), axis=1), 1.0)


def test_l2_normalize_handles_zero_vector():
    x = np.zeros((2, 4))
    assert np.all(np.isfinite(l2_normalize(x)))


def test_rank_by_similarity_finds_nearest():
    corpus = np.eye(5, dtype=np.float32)
    query = np.array([[1.0, 0.1, 0, 0, 0]], dtype=np.float32)
    idx = rank_by_similarity(query, corpus, k=2)
    assert idx[0, 0] == 0
    assert idx[0, 1] == 1


def test_rank_by_similarity_excludes_self():
    """**这条测试防的是最常见的假成绩。**

    库里就含有查询本身时，不剔除会让 HitRate@1 恒等于 1。
    """
    corpus = RNG.standard_normal((20, 8)).astype(np.float32)
    ids = np.arange(20)
    idx = rank_by_similarity(corpus, corpus, k=3, query_ids=ids, corpus_ids=ids)
    for i in range(20):
        assert i not in idx[i], f"查询 {i} 检索到了自己"


def test_rank_by_similarity_without_ids_returns_self_first():
    """反证：不传 ids 时，自己必然排第一 —— 说明剔除逻辑真的在起作用。"""
    corpus = RNG.standard_normal((10, 6)).astype(np.float32)
    idx = rank_by_similarity(corpus, corpus, k=1)
    assert idx[:, 0].tolist() == list(range(10))


# --------------------------------------------------------------------------------------
# 相关性构造
# --------------------------------------------------------------------------------------

def test_shared_tag_counts_is_intersection_size():
    q = np.array([[1, 1, 1, 0, 0]])
    c = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [0, 0, 0, 1, 1]])
    shared = shared_tag_counts(q, c)
    assert shared[0].tolist() == [3.0, 2.0, 0.0]


def test_build_relevance_applies_min_shared_threshold():
    shared = np.array([[5.0, 3.0, 2.0, 0.0]])
    ranked_idx = np.array([[0, 1, 2, 3]])
    rel, gain, n_rel = build_relevance(ranked_idx, shared, min_shared=3)
    assert rel[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert gain[0].tolist() == [5.0, 3.0, 2.0, 0.0]
    assert n_rel[0] == 2.0


# --------------------------------------------------------------------------------------
# HitRate vs Recall：本模块存在的理由
# --------------------------------------------------------------------------------------

def test_hit_rate_and_recall_are_different_things():
    """一个查询，全库 100 个相关项，前 10 命中 1 个。

    - HitRate@10 = 1.0（"命中了吗？" → 命中了）
    - Recall@10  = 0.01（"召回了多少？" → 百分之一）

    两个都对，但回答的是不同问题。混用就是耍流氓。
    """
    ranked_rel = np.zeros((1, 10))
    ranked_rel[0, 0] = 1.0
    n_relevant = np.array([100.0])

    assert hit_rate_at_k(ranked_rel, 10) == pytest.approx(1.0)
    assert recall_at_k(ranked_rel, n_relevant, 10) == pytest.approx(0.01)
    assert precision_at_k(ranked_rel, 10) == pytest.approx(0.1)


def test_perfect_ranking_metrics():
    ranked_rel = np.ones((4, 10))
    n_relevant = np.full(4, 10.0)
    assert hit_rate_at_k(ranked_rel, 1) == pytest.approx(1.0)
    assert precision_at_k(ranked_rel, 10) == pytest.approx(1.0)
    assert recall_at_k(ranked_rel, n_relevant, 10) == pytest.approx(1.0)
    assert average_precision_at_k(ranked_rel, n_relevant, 10) == pytest.approx(1.0)


def test_empty_ranking_metrics_are_zero():
    ranked_rel = np.zeros((3, 10))
    n_relevant = np.full(3, 5.0)
    assert hit_rate_at_k(ranked_rel, 10) == pytest.approx(0.0)
    assert recall_at_k(ranked_rel, n_relevant, 10) == pytest.approx(0.0)
    assert average_precision_at_k(ranked_rel, n_relevant, 10) == pytest.approx(0.0)


def test_hit_rate_is_monotone_in_k():
    ranked_rel = np.zeros((5, 20))
    ranked_rel[:, 7] = 1.0
    assert hit_rate_at_k(ranked_rel, 5) == pytest.approx(0.0)
    assert hit_rate_at_k(ranked_rel, 10) == pytest.approx(1.0)


def test_average_precision_rewards_early_hits():
    """同样命中 2 个，排在前面的 AP 必须更高。"""
    early = np.array([[1.0, 1.0, 0.0, 0.0, 0.0]])
    late = np.array([[0.0, 0.0, 0.0, 1.0, 1.0]])
    n_rel = np.array([2.0])
    assert average_precision_at_k(early, n_rel, 5) > average_precision_at_k(late, n_rel, 5)


def test_recall_skips_queries_without_relevant_items():
    ranked_rel = np.zeros((2, 5))
    ranked_rel[0, 0] = 1.0
    n_relevant = np.array([1.0, 0.0])  # 第二个查询全库无相关项
    assert recall_at_k(ranked_rel, n_relevant, 5) == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# NDCG
# --------------------------------------------------------------------------------------

def test_ndcg_is_one_for_ideal_ordering():
    gain = np.array([[5.0, 4.0, 3.0, 1.0, 0.0]])
    assert ndcg_at_k(gain, 5, ideal_gain=gain) == pytest.approx(1.0)


def test_ndcg_penalizes_bad_ordering():
    ideal = np.array([[5.0, 4.0, 3.0, 1.0, 0.0]])
    actual = np.array([[0.0, 1.0, 3.0, 4.0, 5.0]])
    assert ndcg_at_k(actual, 5, ideal_gain=ideal) < 0.8


def test_ndcg_without_ideal_overestimates():
    """不传全库理想增益时 NDCG 会系统性偏高 —— 这是常见误用，测试把它钉住。"""
    ideal = np.array([[9.0, 8.0, 7.0, 6.0, 5.0]])
    actual = np.array([[1.0, 1.0, 1.0, 1.0, 1.0]])
    lax = ndcg_at_k(actual, 5)                       # 自身降序当理想 → 1.0
    strict = ndcg_at_k(actual, 5, ideal_gain=ideal)  # 真实理想 → 低得多
    assert lax == pytest.approx(1.0)
    assert strict < 0.2


def test_ideal_gains_returns_top_k_descending():
    shared = np.array([[1.0, 9.0, 3.0, 7.0, 0.0]])
    ig = ideal_gains(shared, k=3)
    assert ig[0].tolist() == [9.0, 7.0, 3.0]


# --------------------------------------------------------------------------------------
# 端到端与工程指标
# --------------------------------------------------------------------------------------

def test_evaluate_retrieval_end_to_end():
    """完整链路：嵌入 → 排序 → 相关性 → 指标。

    刻意让嵌入与标签强相关（同簇的歌标签也相同），指标应当明显好于随机。
    """
    n_corpus, n_tags, dim = 200, 10, 16
    cluster = RNG.integers(0, 5, size=n_corpus)

    centers = RNG.standard_normal((5, dim)).astype(np.float32) * 3
    emb = (centers[cluster] + RNG.standard_normal((n_corpus, dim)).astype(np.float32) * 0.3)

    tags = np.zeros((n_corpus, n_tags))
    for i, c in enumerate(cluster):
        tags[i, c] = 1
        tags[i, 5 + c] = 1
        tags[i, (c + 1) % 5] = 1  # 保证共享数能到 3

    ids = np.arange(n_corpus)
    ranked_idx = rank_by_similarity(emb, emb, k=50, query_ids=ids, corpus_ids=ids)
    shared = shared_tag_counts(tags, tags)
    np.fill_diagonal(shared, 0)
    rel, gain, n_rel = build_relevance(ranked_idx, shared, min_shared=3)

    scores = evaluate_retrieval(rel, n_rel, gain, ideal_gains(shared, 50))
    assert scores.n_queries == n_corpus
    assert scores.hit_rate[1] > 0.9
    assert scores.precision[10] > 0.9
    assert 0.0 <= scores.ndcg[10] <= 1.0


def test_evaluate_retrieval_drops_ks_beyond_depth():
    rel = np.ones((3, 10))
    n_rel = np.full(3, 10.0)
    scores = evaluate_retrieval(rel, n_rel, ks=(1, 5, 10, 50))
    assert set(scores.hit_rate) == {1, 5, 10}


def test_evaluate_retrieval_raises_when_all_ks_too_large():
    with pytest.raises(ValueError):
        evaluate_retrieval(np.ones((2, 3)), np.full(2, 3.0), ks=(10, 50))


def test_jaccard_at_k():
    q = np.array([[1, 1, 0, 0]])
    c = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
    ranked = np.array([[0, 1]])
    # 第一个 Jaccard = 2/2 = 1.0，第二个 = 1/2 = 0.5 → 均值 0.75
    assert jaccard_at_k(q, c, ranked, k=2) == pytest.approx(0.75)


def test_index_recall_measures_overlap_with_exact():
    exact = np.array([[0, 1, 2, 3]])
    approx = np.array([[0, 1, 9, 3]])
    assert index_recall(approx, exact, k=4) == pytest.approx(0.75)
    assert index_recall(exact, exact, k=4) == pytest.approx(1.0)
