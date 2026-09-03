"""检索索引的单测。

重点测**排除自身**和**近似索引的召回定义** —— 这两处出错都不会报异常，
只会让指标虚高：前者凭空多一个命中，后者把"没搜到"当成"搜到了别的"。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval.retrieval import index_recall
from src.retrieval.index import ExactIndex, IVFIndex


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    return rng.normal(size=(200, 32)).astype(np.float32)


def test_exact_search_excludes_self(corpus):
    """查询就在库里时，第一名永远是自己 —— 不排除会让 Recall 凭空多一个命中。"""
    idx, _ = ExactIndex(corpus).search(corpus, k=5)
    for i in range(len(corpus)):
        assert i not in idx[i], f"第 {i} 条检索到了自己"


def test_exact_search_can_include_self(corpus):
    idx, _ = ExactIndex(corpus).search(corpus, k=1, exclude_self=False)
    assert (idx[:, 0] == np.arange(len(corpus))).all(), "不排除时第一名必须是自己"


def test_exact_finds_planted_duplicate():
    """给第 0 条种一个几乎相同的副本，它必须排第一。"""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 16)).astype(np.float32)
    X[7] = X[0] + 1e-4                      # 近似副本
    idx, _ = ExactIndex(X).search(X[:1], k=1)
    assert idx[0, 0] == 7


def test_exact_results_are_sorted_by_similarity(corpus):
    ex = ExactIndex(corpus)
    idx, _ = ex.search(corpus[:5], k=10)
    for q in range(5):
        sims = ex.emb[idx[q]] @ ex.emb[q]
        assert (np.diff(sims) <= 1e-6).all(), "结果必须按相似度降序"


def test_cosine_is_scale_invariant(corpus):
    """余弦相似度与向量长度无关 —— 放大 10 倍不该改变排序。"""
    a, _ = ExactIndex(corpus).search(corpus[:5], k=10)
    b, _ = ExactIndex(corpus * 10).search(corpus[:5] * 10, k=10)
    assert (a == b).all()


def test_ivf_recall_is_reported_not_assumed(corpus):
    """近似索引的召回必须**实测**。

    n_probe=1 时召回明显低于 1，这是近似方法的固有代价；
    不报这个数字就等于把"快了几倍"说成免费的。
    """
    exact, _ = ExactIndex(corpus).search(corpus, k=10)
    ivf = IVFIndex(corpus, n_lists=16)
    r1 = index_recall(ivf.search(corpus, k=10, n_probe=1)[0], exact, 10)
    r_all = index_recall(ivf.search(corpus, k=10, n_probe=16)[0], exact, 10)
    assert 0.0 < r1 < 1.0, f"n_probe=1 的召回应当明显不足，得到 {r1}"
    assert r_all > r1, "搜更多桶必须让召回上升"


def test_ivf_probing_all_lists_matches_exact(corpus):
    """搜遍所有桶时，近似必须退化成精确。做不到就说明分桶或候选合并有 bug。"""
    exact, _ = ExactIndex(corpus).search(corpus, k=5)
    ivf = IVFIndex(corpus, n_lists=8)
    approx, _ = ivf.search(corpus, k=5, n_probe=8)
    assert index_recall(approx, exact, 5) == pytest.approx(1.0)


def test_ivf_compares_fewer_vectors(corpus):
    """近似的全部意义就是少算 —— 少算不了就没有存在价值。"""
    _, st = IVFIndex(corpus, n_lists=16).search(corpus, k=10, n_probe=2)
    assert st.n_compared < len(corpus) ** 2 * 0.5


def test_ivf_excludes_self(corpus):
    idx, _ = IVFIndex(corpus, n_lists=8).search(corpus, k=5, n_probe=4)
    for i in range(len(corpus)):
        assert i not in idx[i]


def test_stats_report_actual_work(corpus):
    _, st = ExactIndex(corpus).search(corpus[:10], k=5)
    assert st.n_compared == 10 * len(corpus)
    assert st.seconds > 0
