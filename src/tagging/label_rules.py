"""训练标签的层级补全。

## 为什么要做

MTG-Jamendo 的标签来自上传者，**正例不完整**：没标的不等于没有。
P9 量出了最直接的证据 —— house **按定义**就是 electronic，
但训练集里只有 **53%** 的 house 曲目被标了 electronic。

补全的想法：训练时把「子 ⇒ 父」关系补上，让模型别再把
"house 曲目也报 electronic" 当成错误来惩罚。

## 两种规则，分开验

- :data:`SEMANTIC_RULES`：只收**按定义成立**的子流派 ⇒ 父流派。跑实验前写死
- :func:`mine_rules`：从训练集共现里挖。会混进**只是相关、并非层级**的规则
  （如 emotional ⇒ piano），用来检验"不加甄别地补全"的代价

## 只补训练集

验证集与测试集**保持原样**。验证集用于早停与阈值搜索、测试集用于报数，
两者若也被补全，就等于改了考卷再说分数涨了。

.. warning::
   **拿补全过的标签去评测一个用补全标签训出来的模型，是循环论证。**
   分数涨了只说明模型学会了我们注入的规则，证明不了它"更对"。
   真正的裁判只能是原始测试标签，或者人工听审。
"""

from __future__ import annotations

import numpy as np

# 跑实验前写死，见 results/P10_预测.md。
# 只收电子乐子流派 ⇒ electronic 这种**按定义成立**的关系；
# dance 排除在外 —— 舞曲大多是电子乐，但不是按定义。
SEMANTIC_RULES: tuple[tuple[str, str], ...] = (
    ("genre---house", "genre---electronic"),
    ("genre---techno", "genre---electronic"),
    ("genre---trance", "genre---electronic"),
    ("genre---triphop", "genre---electronic"),
    ("genre---downtempo", "genre---electronic"),
)


def mine_rules(Y: np.ndarray, names: list[str] | tuple[str, ...], min_conf: float = 0.5,
               max_reverse: float = 0.3, min_support: int = 20) -> tuple[tuple[str, str], ...]:
    """从共现里挖「a ⇒ b」：P(b|a) ≥ min_conf 且 P(a|b) < max_reverse。

    要求**不对称**才算层级：bass 与 drums 互相高度共现（0.68 / 0.60），
    那是两件常一起出现的乐器，不是谁包含谁。

    **只能传训练集进来** —— 用测试集挖规则就是标签泄漏。
    """
    B = np.asarray(Y) > 0.5
    n = B.sum(axis=0).astype(float)
    co = B.T.astype(float) @ B.astype(float)
    out = []
    for a in range(len(names)):
        if n[a] < min_support:
            continue
        for b in range(len(names)):
            if a == b or n[b] == 0:
                continue
            if co[a, b] / n[a] >= min_conf and co[a, b] / n[b] < max_reverse:
                out.append((names[a], names[b]))
    return tuple(out)


def propagate(Y: np.ndarray, names: list[str] | tuple[str, ...],
              rules: tuple[tuple[str, str], ...]) -> np.ndarray:
    """按规则补全正例，返回**新数组**（不改原数组）。

    - 只加不减：已有的正例一个都不会被清掉
    - 做传递闭包：a ⇒ b ⇒ c 时，只有 a 的曲目会同时补上 b 和 c
    - 规则里出现词表没有的标签时**直接报错**。拼错一个标签名若被静默跳过，
      实验就会在"没补全"的状态下跑完，而结果看上去完全正常
    """
    idx = {t: i for i, t in enumerate(names)}
    missing = sorted({t for r in rules for t in r if t not in idx})
    if missing:
        raise ValueError(f"规则引用了词表里没有的标签：{missing}")

    out = np.array(Y, dtype=np.float32, copy=True)
    pairs = [(idx[a], idx[b]) for a, b in rules]
    changed = True
    while changed:                      # 只做"加"，单调，必然收敛（有环也不会死循环）
        changed = False
        for a, b in pairs:
            add = (out[:, a] > 0.5) & (out[:, b] < 0.5)
            if add.any():
                out[add, b] = 1.0
                changed = True
    return out
