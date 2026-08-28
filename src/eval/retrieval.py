"""相似音乐检索评测。

**一个必须先讲清楚的坑：文献里的 "Recall@K" 至少有两种互不兼容的定义。**

- **命中率口径（Hit Rate）**：前 K 个里只要出现 ≥1 个相关项就记 1。
  文本→音乐检索里每个查询只有唯一正确答案，用的就是这个（所以那些论文里
  R@10 才十几个点也不奇怪）。
- **真召回口径（true Recall）**：``前K个里的相关数 / 全库相关总数``。
  这是 IR 教科书定义。本项目的代理真值（共享 ≥3 个标签）会让每个查询有成百上千个
  相关项，分母巨大，K=10 时数值必然极小。

本模块**两个都算、分开命名**，绝不混用：:func:`hit_rate_at_k` 与 :func:`recall_at_k`。
结果表主报 **HitRate@K + Precision@K + NDCG@K**，把 true Recall 作为补充。

**第二个必须声明的事：这里的"相关"是代理真值，不是人类听感相似度。**
两首歌共享标签少但听起来很像时会被判为"不相关"，从而系统性低估性能。
所以 P5 还要求做 20 组人工听感抽查，用来校准这个代理指标的可信度。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------------------
# 构造排序与相关性
# --------------------------------------------------------------------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, eps)


def rank_by_similarity(
    query_emb: np.ndarray,
    corpus_emb: np.ndarray,
    k: int,
    query_ids: np.ndarray | None = None,
    corpus_ids: np.ndarray | None = None,
) -> np.ndarray:
    """余弦相似度暴力检索，返回 ``(n_queries, k)`` 的库内下标。

    这是**精确基准**（对应 FAISS 的 IndexFlatIP）。HNSW 等近似索引的召回损失
    要以它为分母来报。

    Args:
        query_ids/corpus_ids: 传了就会把"查询自己"从候选中剔除。
            **一定要传** —— 忘了剔除会让 HitRate@1 恒等于 1，是这类实验最常见的假成绩。
    """
    q = l2_normalize(np.asarray(query_emb, dtype=np.float32))
    c = l2_normalize(np.asarray(corpus_emb, dtype=np.float32))
    sim = q @ c.T

    if query_ids is not None and corpus_ids is not None:
        # 用广播找到 self 位置，置为 -inf
        self_mask = np.asarray(query_ids)[:, None] == np.asarray(corpus_ids)[None, :]
        sim[self_mask] = -np.inf

    k = min(k, sim.shape[1])
    # argpartition 取前 k 再排序，比全排序快一个量级
    idx = np.argpartition(-sim, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(sim, idx, axis=1), axis=1)
    return np.take_along_axis(idx, order, axis=1)


def shared_tag_counts(query_tags: np.ndarray, corpus_tags: np.ndarray) -> np.ndarray:
    """``(n_queries, n_corpus)`` 的共享标签数矩阵。就是 0/1 标签矩阵相乘。"""
    return (np.asarray(query_tags, dtype=np.float32) @ np.asarray(corpus_tags, dtype=np.float32).T)


def build_relevance(
    ranked_idx: np.ndarray,
    shared: np.ndarray,
    min_shared: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从共享标签数矩阵构造评测所需的三件套。

    Returns:
        - ``ranked_rel`` ``(n_q, K)`` 0/1，前 K 个结果是否相关
        - ``ranked_gain`` ``(n_q, K)`` 分级增益（= 共享标签数），给 NDCG 用
        - ``n_relevant`` ``(n_q,)`` 全库相关总数，给 true Recall 用
    """
    ranked_gain = np.take_along_axis(shared, ranked_idx, axis=1)
    ranked_rel = (ranked_gain >= min_shared).astype(np.float64)
    n_relevant = (shared >= min_shared).sum(axis=1).astype(np.float64)
    return ranked_rel, ranked_gain, n_relevant


def jaccard_at_k(query_tags: np.ndarray, corpus_tags: np.ndarray, ranked_idx: np.ndarray, k: int) -> float:
    """前 k 个检索结果与查询的**平均标签 Jaccard 相似度**。

    辅助指标，但很直观：0.4 意味着检索出来的歌平均有四成标签和查询重合。
    """
    q = np.asarray(query_tags, dtype=bool)
    c = np.asarray(corpus_tags, dtype=bool)
    vals = []
    for i in range(q.shape[0]):
        cand = c[ranked_idx[i, :k]]
        inter = (cand & q[i]).sum(axis=1)
        union = (cand | q[i]).sum(axis=1)
        vals.append(np.mean(np.where(union > 0, inter / np.maximum(union, 1), 0.0)))
    return float(np.mean(vals))


# --------------------------------------------------------------------------------------
# 指标
# --------------------------------------------------------------------------------------

def hit_rate_at_k(ranked_rel: np.ndarray, k: int) -> float:
    """前 K 个里**至少有一个**相关项的查询比例。文本→音乐检索文献常用口径。"""
    return float(np.mean(ranked_rel[:, :k].max(axis=1) > 0))


def recall_at_k(ranked_rel: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """真召回：``前K个里的相关数 / 全库相关总数``。无相关项的查询被剔除。"""
    n_relevant = np.asarray(n_relevant, dtype=np.float64)
    valid = n_relevant > 0
    if not valid.any():
        return float("nan")
    hits = ranked_rel[:, :k].sum(axis=1)
    return float(np.mean(hits[valid] / n_relevant[valid]))


def precision_at_k(ranked_rel: np.ndarray, k: int) -> float:
    """前 K 个里相关项的比例。K 小时比 true Recall 直观得多。"""
    return float(np.mean(ranked_rel[:, :k].sum(axis=1) / k))


def average_precision_at_k(ranked_rel: np.ndarray, n_relevant: np.ndarray, k: int) -> float:
    """mAP@K：不仅看在不在前 K，还看排得多靠前。

    分母取 ``min(全库相关数, K)``，即"理想情况下前 K 最多能命中多少"，
    这是 TREC 风格的做法，避免相关项极多时 AP 被压到接近 0。
    """
    n_relevant = np.asarray(n_relevant, dtype=np.float64)
    rel = ranked_rel[:, :k]
    cum_hits = np.cumsum(rel, axis=1)
    ranks = np.arange(1, rel.shape[1] + 1)[None, :]
    precisions = cum_hits / ranks
    denom = np.minimum(n_relevant, k)
    valid = denom > 0
    if not valid.any():
        return float("nan")
    ap = np.sum(precisions * rel, axis=1) / np.maximum(denom, 1e-12)
    return float(np.mean(ap[valid]))


def ndcg_at_k(
    ranked_gain: np.ndarray,
    k: int,
    ideal_gain: np.ndarray | None = None,
) -> float:
    """NDCG@K，支持分级相关性（共享 5 个标签比共享 3 个更相关）。

    Args:
        ranked_gain: ``(n_q, ≥k)`` 检索结果的增益。
        ideal_gain: ``(n_q, ≥k)`` **全库**最优排序的前 k 个增益。
            不传则退化为"用检索结果自身降序排"作为理想 —— 那是 NDCG 的常见误用，
            会系统性高估。**正式报数必须传全库理想增益。**
    """
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(ranked_gain[:, :k] * discount[None, :], axis=1)
    if ideal_gain is None:
        ideal = -np.sort(-ranked_gain[:, :k], axis=1)
    else:
        ideal = ideal_gain[:, :k]
    idcg = np.sum(ideal * discount[None, :], axis=1)
    valid = idcg > 0
    if not valid.any():
        return float("nan")
    return float(np.mean(dcg[valid] / idcg[valid]))


def ideal_gains(shared: np.ndarray, k: int) -> np.ndarray:
    """全库最优排序的前 k 个增益，给 :func:`ndcg_at_k` 当分母。"""
    k = min(k, shared.shape[1])
    part = np.partition(-shared, kth=k - 1, axis=1)[:, :k]
    return -np.sort(part, axis=1)


@dataclass
class RetrievalScores:
    hit_rate: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    map: dict[int, float]
    ndcg: dict[int, float]
    n_queries: int
    mean_n_relevant: float

    def to_row(self, name: str, ks: tuple[int, ...] = (1, 5, 10, 50)) -> dict:
        row: dict = {"表征": name}
        for k in ks:
            row[f"HitRate@{k}"] = self.hit_rate.get(k, float("nan"))
        row["P@10"] = self.precision.get(10, float("nan"))
        row["mAP@10"] = self.map.get(10, float("nan"))
        row["NDCG@10"] = self.ndcg.get(10, float("nan"))
        row["Recall@50"] = self.recall.get(50, float("nan"))
        return row


def evaluate_retrieval(
    ranked_rel: np.ndarray,
    n_relevant: np.ndarray,
    ranked_gain: np.ndarray | None = None,
    ideal_gain: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5, 10, 50),
) -> RetrievalScores:
    """一次算全所有检索指标。"""
    k_max = ranked_rel.shape[1]
    ks = tuple(k for k in ks if k <= k_max)
    if not ks:
        raise ValueError(f"所有 K 都超过了检索深度 {k_max}")

    gain = ranked_rel if ranked_gain is None else ranked_gain
    return RetrievalScores(
        hit_rate={k: hit_rate_at_k(ranked_rel, k) for k in ks},
        recall={k: recall_at_k(ranked_rel, n_relevant, k) for k in ks},
        precision={k: precision_at_k(ranked_rel, k) for k in ks},
        map={k: average_precision_at_k(ranked_rel, n_relevant, k) for k in ks},
        ndcg={k: ndcg_at_k(gain, k, ideal_gain) for k in ks},
        n_queries=int(ranked_rel.shape[0]),
        mean_n_relevant=float(np.mean(n_relevant)),
    )


def index_recall(approx_idx: np.ndarray, exact_idx: np.ndarray, k: int) -> float:
    """近似索引（HNSW/IVF）相对精确检索（Flat）的召回率。

    P5 要报的"HNSW 损失了多少召回换来多少倍加速"，分子就是这个。
    """
    vals = []
    for a, e in zip(approx_idx[:, :k], exact_idx[:, :k], strict=False):
        vals.append(len(set(a.tolist()) & set(e.tolist())) / k)
    return float(np.mean(vals))
