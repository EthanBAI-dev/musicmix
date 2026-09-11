"""标签错误分析的前提与权重元数据。

P9 整份分析建立在一个前提上：**随机打分的 AP 约等于正例率**，
因此提升倍数（AP ÷ 正例率）才跨标签可比。这个前提若不成立，
"按提升倍数判断难度"就没有依据 —— 所以先把它测死。
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.eval.tagging import per_tag_average_precision
from src.tagging.train import TrainResult, save_result

FEATS = {"backbone": "m-a-p/MERT-v1-95M", "layer": 6, "segments": 4,
         "use_segments": "", "frame_stride": 5}


def test_random_scores_have_lift_near_one():
    """随机打分时，各正例率下的提升倍数都应约等于 1。

    这是 P9 的方法学前提：它说明原始 AP 的高低**主要由正例率决定**，
    直接按 AP 排序只是在排稀有度。
    """
    rng = np.random.default_rng(0)
    prev = np.array([0.01, 0.05, 0.10, 0.30, 0.50])
    y = (rng.random((50_000, len(prev))) < prev).astype(float)
    ap = per_tag_average_precision(y, rng.random(y.shape))
    lift = ap / y.mean(axis=0)
    assert np.all((lift > 0.75) & (lift < 1.3)), f"随机打分的提升倍数应≈1，得到 {lift}"


def test_raw_ap_ranks_rarity_not_difficulty():
    """两个标签**同样被随机打分**（难度相同），原始 AP 仍会因稀有度差出一大截。"""
    rng = np.random.default_rng(1)
    y = np.stack([rng.random(50_000) < 0.02, rng.random(50_000) < 0.40], axis=1).astype(float)
    ap = per_tag_average_precision(y, rng.random(y.shape))
    assert ap[1] > ap[0] * 5, "难度相同，原始 AP 却应被正例率拉开"
    lift = ap / y.mean(axis=0)
    assert abs(lift[0] - lift[1]) < 0.3, "换成提升倍数后，两者应当接近"


def _result(n_tags: int = 3) -> TrainResult:
    return TrainResult(level="L2", config={"pooling": "mean", "loss": "bce"}, best_epoch=3,
                       thresholds=np.full(n_tags, 0.5), state_dict={"w": torch.zeros(2)})


def test_checkpoint_records_which_features_it_was_trained_on(tmp_path):
    """权重必须记录自己吃的是哪份特征。

    对应一个真实缺陷：best_model.pt 的 config 里只有 pooling 和 loss，
    说不出是用第 6 层 4 段训的。任何层都是 768 维，配错特征加载**不会报错**，
    只会静默给出错误的预测。
    """
    out = tmp_path / "m.json"
    save_result(_result(), out, tag_names=["a", "b", "c"], save_model=True, features=FEATS)
    ck = torch.load(out.with_suffix(".pt"), map_location="cpu", weights_only=False)
    assert ck["features"] == FEATS
    assert json.loads(out.read_text(encoding="utf-8"))["features"] == FEATS


def test_missing_features_do_not_crash(tmp_path):
    """旧调用方不传 features 时照常工作，字段为空 dict 而不是缺失。"""
    out = tmp_path / "m.json"
    save_result(_result(), out, tag_names=["a", "b", "c"], save_model=True)
    ck = torch.load(out.with_suffix(".pt"), map_location="cpu", weights_only=False)
    assert ck["features"] == {}


def test_no_checkpoint_unless_asked(tmp_path):
    """默认不存权重 —— 多种子实验会堆出上百个没用的 checkpoint。"""
    out = tmp_path / "m.json"
    save_result(_result(), out, tag_names=["a", "b", "c"], features=FEATS)
    assert out.exists() and not out.with_suffix(".pt").exists()


def test_threshold_and_tag_count_must_match(tmp_path):
    """阈值与标签名用 strict=True 配对 —— 数量不一致必须当场报错，而不是错位写入。"""
    with pytest.raises(ValueError):
        save_result(_result(n_tags=3), tmp_path / "m.json", tag_names=["a", "b"])
