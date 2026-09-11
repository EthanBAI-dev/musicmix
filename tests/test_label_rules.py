"""层级补全的单测。

补全出错**不会报异常**：规则写反、漏补、误删正例，模型照训、分数照出，
只是实验测的不再是它声称要测的东西。所以逐条断言语义。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tagging.label_rules import SEMANTIC_RULES, mine_rules, propagate

N = ["c", "p", "x", "y", "z"]


def test_child_gets_parent():
    Y = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
    assert propagate(Y, N, (("c", "p"),))[0].tolist() == [1, 1, 0, 0, 0]


def test_never_removes_positives():
    Y = np.array([[0, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=np.float32)
    out = propagate(Y, N, (("c", "p"),))
    assert (out >= Y).all(), "补全只能加正例，不能清掉已有的"


def test_rows_without_child_untouched():
    Y = np.array([[0, 0, 1, 0, 0]], dtype=np.float32)
    assert (propagate(Y, N, (("c", "p"),)) == Y).all()


def test_transitive_closure():
    """a ⇒ b ⇒ c：只有 a 的曲目要同时补上 b 和 c。"""
    Y = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
    out = propagate(Y, N, (("c", "p"), ("p", "x")))
    assert out[0].tolist() == [1, 1, 1, 0, 0]


def test_cycle_terminates():
    Y = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
    out = propagate(Y, N, (("c", "p"), ("p", "c")))
    assert out[0].tolist() == [1, 1, 0, 0, 0]


def test_unknown_tag_raises():
    """拼错标签名若被静默跳过，实验会在"没补全"的状态下跑完，结果还看着正常。"""
    with pytest.raises(ValueError, match="没有的标签"):
        propagate(np.zeros((1, 5), np.float32), N, (("c", "electornic"),))


def test_input_not_mutated():
    Y = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)
    before = Y.copy()
    propagate(Y, N, (("c", "p"),))
    assert (Y == before).all()


def _planted() -> np.ndarray:
    """c ⇒ p 是真层级；x、y 互相重叠；z 太少。"""
    Y = np.zeros((1000, 5), dtype=np.float32)
    Y[0:100, 0] = 1; Y[0:100, 1] = 1          # c 全部带 p
    Y[100:600, 1] = 1                          # p 另有 500 首单独出现 → 反向弱
    Y[600:800, 2] = 1; Y[600:800, 3] = 1       # x、y 总是一起 → 对称
    Y[900:905, 4] = 1; Y[900:905, 1] = 1       # z 只有 5 首
    return Y


def test_mining_finds_planted_hierarchy_only():
    assert set(mine_rules(_planted(), N)) == {("c", "p")}


def test_mining_excludes_symmetric_overlap():
    """bass 与 drums 那种互相高度共现是两件乐器常一起出现，不是层级。"""
    rules = set(mine_rules(_planted(), N))
    assert ("x", "y") not in rules and ("y", "x") not in rules


def test_mining_respects_min_support():
    assert ("z", "p") not in set(mine_rules(_planted(), N, min_support=20))
    assert ("z", "p") in set(mine_rules(_planted(), N, min_support=5))


def test_semantic_rules_are_well_formed():
    """语义规则是跑实验前写死的：全部指向 electronic，子流派互不重复，且不自指。"""
    parents = {b for _, b in SEMANTIC_RULES}
    children = [a for a, _ in SEMANTIC_RULES]
    assert parents == {"genre---electronic"}
    assert len(children) == len(set(children))
    assert all(a != b for a, b in SEMANTIC_RULES)
    assert "genre---dance" not in children, "dance 按定义并非电子乐，预注册时已排除"
