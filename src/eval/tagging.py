"""多标签音乐标签评测。

这是 P4 的记分牌。几个必须一次性说清、之后再也不许含糊的口径问题：

**1. 这是多标签（multi-label），不是多分类（multi-class）。**
一首歌可以同时是 rock + energetic + guitar，标签互不排斥。所以输出层是 Sigmoid、
损失是逐标签 BCE，而 **Accuracy 毫无意义**（全预测 0 也能有 95% 准确率）。

**2. mAP 与阈值无关，F1 强依赖阈值。**
mAP 看的是排序质量；F1 要先把概率二值化。因此"调完阈值后 mAP 一字不变、
Macro-F1 涨了 5 个点"是完全正常的现象 —— 这正是 P4-L4 要做的那张对比图。

**3. Macro 与 Micro 回答的是不同问题。**
- Macro：每个标签单独算再等权平均 → 稀有标签和热门标签同等重要 → **看长尾**
- Micro：全局汇总 TP/FP/FN 算一次 → 被热门标签主导 → **看整体**
本项目主报 **Macro**，因为 183 个标签里大部分是长尾，改进只可能在 Macro 上体现。

**4. 退化标签的处理是统一的。**
测试集里可能有标签一个正样本都没有（AUC/AP 无定义）。本模块用**同一个 valid 掩码**
（至少 1 个正样本且至少 1 个负样本）过滤所有 macro 指标，并把 ``n_valid_tags``
一起返回。**结果表里必须带上这个数**，否则两次实验的 macro 值可能根本不可比。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT_THRESHOLD_GRID = np.round(np.arange(0.01, 1.00, 0.01), 4)


def valid_tag_mask(y_true: np.ndarray) -> np.ndarray:
    """可评测的标签掩码：至少 1 个正样本且至少 1 个负样本。"""
    y_true = np.asarray(y_true)
    pos = y_true.sum(axis=0)
    return (pos > 0) & (pos < y_true.shape[0])


# --------------------------------------------------------------------------------------
# 与阈值无关的排序类指标
# --------------------------------------------------------------------------------------

def macro_roc_auc(y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray | None = None) -> float:
    """逐标签 ROC-AUC 再等权平均。

    Warning:
        在极度不平衡的数据上 ROC-AUC 会**系统性虚高**（负样本太多，随便排排也好看）。
        报出来只为与旧文献对齐，**判断模型好坏请看 mAP**。
    """
    from sklearn.metrics import roc_auc_score

    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    mask = valid_tag_mask(y_true) if mask is None else mask
    if not mask.any():
        return float("nan")
    return float(roc_auc_score(y_true[:, mask], y_score[:, mask], average="macro"))


def macro_average_precision(
    y_true: np.ndarray, y_score: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """mAP = 逐标签 AP（PR 曲线下面积）再等权平均。**本任务主指标之一。**"""
    from sklearn.metrics import average_precision_score

    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    mask = valid_tag_mask(y_true) if mask is None else mask
    if not mask.any():
        return float("nan")
    return float(average_precision_score(y_true[:, mask], y_score[:, mask], average="macro"))


def per_tag_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> np.ndarray:
    """逐标签 AP，退化标签给 NaN。用来做"最难的 20 个标签"错误分析。"""
    from sklearn.metrics import average_precision_score

    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    out = np.full(y_true.shape[1], np.nan)
    mask = valid_tag_mask(y_true)
    for j in np.flatnonzero(mask):
        out[j] = average_precision_score(y_true[:, j], y_score[:, j])
    return out


# --------------------------------------------------------------------------------------
# 依赖阈值的 F1 类指标
# --------------------------------------------------------------------------------------

def _counts_at(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回逐标签的 (tp, fp, fn)。"""
    tp = np.sum(y_true * y_pred, axis=0)
    fp = np.sum((1 - y_true) * y_pred, axis=0)
    fn = np.sum(y_true * (1 - y_pred), axis=0)
    return tp, fp, fn


def _f1_from_counts(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denom = 2 * tp + fp + fn
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def per_tag_f1(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: float | np.ndarray = 0.5
) -> np.ndarray:
    """逐标签 F1。``thresholds`` 可以是标量，也可以是长度为 n_tags 的向量。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    th = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (y_true.shape[1],))
    y_pred = (y_score >= th[None, :]).astype(np.float64)
    return _f1_from_counts(*_counts_at(y_true, y_pred))


def macro_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    mask: np.ndarray | None = None,
) -> float:
    y_true = np.asarray(y_true)
    mask = valid_tag_mask(y_true) if mask is None else mask
    f1 = per_tag_f1(y_true, y_score, thresholds)
    return float(np.mean(f1[mask])) if mask.any() else float("nan")


def micro_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
    mask: np.ndarray | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    mask = valid_tag_mask(y_true) if mask is None else mask
    th = np.broadcast_to(np.asarray(thresholds, dtype=np.float64), (y_true.shape[1],))
    y_pred = (y_score >= th[None, :]).astype(np.float64)
    tp, fp, fn = _counts_at(y_true[:, mask], y_pred[:, mask])
    return float(_f1_from_counts(tp.sum(), fp.sum(), fn.sum()))


def tune_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: np.ndarray = DEFAULT_THRESHOLD_GRID,
    default: float = 0.5,
) -> np.ndarray:
    """**在验证集上**为每个标签独立搜索使该标签 F1 最大的阈值。

    这是 P4-L4，也是本项目性价比最高的一招：稀有标签的预测概率可能永远
    到不了 0.5，用默认阈值等于直接放弃它们，而 Macro-F1 对此极其敏感。

    Args:
        grid: 候选阈值。默认 0.01~0.99 步长 0.01。
        default: 退化标签（无正样本）用的兜底阈值。

    Returns:
        ``(n_tags,)`` 阈值向量。

    Warning:
        **只能在验证集上调，调完固定住再上测试集。** 在测试集上调阈值等于作弊，
        数字会虚高且不可复现。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_tags = y_true.shape[1]

    best_f1 = np.full(n_tags, -1.0)
    best_th = np.full(n_tags, default, dtype=np.float64)

    # 对阈值循环、对 (样本 × 标签) 向量化：183 标签 × 1.1 万样本 × 99 阈值 ≈ 1 秒
    for t in grid:
        y_pred = (y_score >= t).astype(np.float64)
        f1 = _f1_from_counts(*_counts_at(y_true, y_pred))
        better = f1 > best_f1
        best_f1[better] = f1[better]
        best_th[better] = t

    mask = valid_tag_mask(y_true)
    best_th[~mask] = default
    return best_th


# --------------------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------------------

@dataclass
class TaggingScores:
    roc_auc: float
    map: float
    macro_f1_default: float
    macro_f1_tuned: float
    micro_f1_default: float
    micro_f1_tuned: float
    n_tags: int
    n_valid_tags: int
    groups: dict[str, TaggingScores] = field(default_factory=dict)

    def to_row(self, name: str) -> dict:
        return {
            "配置": name,
            "ROC-AUC": self.roc_auc,
            "mAP": self.map,
            "Macro-F1@0.5": self.macro_f1_default,
            "Macro-F1@tuned": self.macro_f1_tuned,
            "Micro-F1@tuned": self.micro_f1_tuned,
            "有效标签数": f"{self.n_valid_tags}/{self.n_tags}",
        }


def evaluate_tagging(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray | None = None,
    tag_groups: dict[str, np.ndarray] | None = None,
) -> TaggingScores:
    """一次算全所有指标。

    Args:
        y_true: ``(n_samples, n_tags)`` 0/1。
        y_score: ``(n_samples, n_tags)`` sigmoid 概率。
        thresholds: :func:`tune_thresholds` **在验证集上**产出的阈值向量。
            传 None 则 tuned 列等于 default 列。
        tag_groups: ``{"genre": idx数组, "instrument": ..., "mood/theme": ...}``。
            传了就会额外给出分组指标 —— **P4-L5 的 Stem-aware 假设检验就靠这个**：
            如果 instrument 组的 Δ 明显大于 mood 组，假设 H2 成立。

    Returns:
        :class:`TaggingScores`
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.shape != y_score.shape:
        raise ValueError(f"形状不一致：y_true={y_true.shape} y_score={y_score.shape}")

    mask = valid_tag_mask(y_true)
    th = np.full(y_true.shape[1], 0.5) if thresholds is None else np.asarray(thresholds)

    scores = TaggingScores(
        roc_auc=macro_roc_auc(y_true, y_score, mask),
        map=macro_average_precision(y_true, y_score, mask),
        macro_f1_default=macro_f1(y_true, y_score, 0.5, mask),
        macro_f1_tuned=macro_f1(y_true, y_score, th, mask),
        micro_f1_default=micro_f1(y_true, y_score, 0.5, mask),
        micro_f1_tuned=micro_f1(y_true, y_score, th, mask),
        n_tags=int(y_true.shape[1]),
        n_valid_tags=int(mask.sum()),
    )

    if tag_groups:
        for gname, idx in tag_groups.items():
            idx = np.asarray(idx)
            sub_th = th[idx] if thresholds is not None else None
            scores.groups[gname] = evaluate_tagging(y_true[:, idx], y_score[:, idx], sub_th)
    return scores


def hardest_tags(
    y_true: np.ndarray, y_score: np.ndarray, tag_names: list[str], k: int = 20
) -> list[tuple[str, float, int]]:
    """AP 最低的 k 个标签，返回 ``(标签名, AP, 正样本数)``。

    做错误分析用：看它们是"样本太少"还是"标签定义本身就模糊"
    （像 ``epic`` / ``background`` 这类主观标签，人标的一致性都很差）。
    """
    ap = per_tag_average_precision(y_true, y_score)
    pos = np.asarray(y_true).sum(axis=0).astype(int)
    order = np.argsort(np.where(np.isnan(ap), np.inf, ap))
    out = []
    for j in order:
        if np.isnan(ap[j]):
            continue
        out.append((tag_names[j], float(ap[j]), int(pos[j])))
        if len(out) >= k:
            break
    return out
