"""多段特征的切片：一旦切错位，模型照样能训，只是永远学不好。

这类错误不会抛异常、不会让 loss 变成 NaN，只会让指标悄悄低一截 ——
和缓存键碰撞、A/B 导出错位属于同一类**静默失败**。
所以这里逐行验证切出来的到底是哪几段，而不只是验证「形状对」。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tagging.dataset import FeatureDataset

N_SEG, FRAMES_PER_SEG, DIM = 4, 3, 2


def _make(tmp_path):
    """造一个 4 段的特征文件，**第 k 段的所有元素都等于 k**。

    这样切片是否正确可以直接从数值读出来，不必依赖形状去推断。
    """
    x = np.repeat(np.arange(N_SEG, dtype=np.float32), FRAMES_PER_SEG)
    x = np.tile(x[:, None], (1, DIM))            # (12, 2)
    p = tmp_path / "feat.npy"
    np.save(p, x)
    return [p], np.zeros((1, 5), dtype=np.float32)


def test_single_segment_picks_that_segment(tmp_path):
    paths, y = _make(tmp_path)
    ds = FeatureDataset(paths, y, n_segments=N_SEG, use_segments=(2,))
    x, _ = ds[0]
    assert x.shape == (FRAMES_PER_SEG, DIM)
    assert (x.numpy() == 2).all(), "取第 2 段，就必须全是 2"


def test_two_segments_keep_order_and_content(tmp_path):
    paths, y = _make(tmp_path)
    ds = FeatureDataset(paths, y, n_segments=N_SEG, use_segments=(0, 2))
    x = ds[0][0].numpy()
    assert x.shape == (2 * FRAMES_PER_SEG, DIM)
    assert (x[:FRAMES_PER_SEG] == 0).all()
    assert (x[FRAMES_PER_SEG:] == 2).all()


def test_none_uses_everything(tmp_path):
    paths, y = _make(tmp_path)
    ds = FeatureDataset(paths, y, n_segments=N_SEG, use_segments=None)
    x = ds[0][0].numpy()
    assert x.shape == (N_SEG * FRAMES_PER_SEG, DIM)
    assert [x[i * FRAMES_PER_SEG, 0] for i in range(N_SEG)] == [0, 1, 2, 3]


def test_selection_order_is_preserved(tmp_path):
    """乱序选段应当按**给定顺序**拼，不是按段号排序 ——
    池化对顺序不敏感，但若将来换成对顺序敏感的头，静默重排就是个坑。"""
    paths, y = _make(tmp_path)
    x = FeatureDataset(paths, y, n_segments=N_SEG, use_segments=(3, 1))[0][0].numpy()
    assert (x[:FRAMES_PER_SEG] == 3).all()
    assert (x[FRAMES_PER_SEG:] == 1).all()


def test_non_divisible_frame_count_raises(tmp_path):
    """帧数除不尽段数 = 缓存不是这个段数提的。宁可炸掉也不能静默切错位。"""
    p = tmp_path / "odd.npy"
    np.save(p, np.zeros((10, DIM), dtype=np.float32))     # 10 除不尽 4
    ds = FeatureDataset([p], np.zeros((1, 5), np.float32),
                        n_segments=N_SEG, use_segments=(0,))
    with pytest.raises(ValueError, match="无法整除"):
        _ = ds[0]


def test_out_of_range_segment_raises_at_construction(tmp_path):
    """越界应当在**构造时**就报，而不是等训练跑到一半才报。"""
    paths, y = _make(tmp_path)
    with pytest.raises(ValueError, match="越界"):
        FeatureDataset(paths, y, n_segments=N_SEG, use_segments=(0, 9))


def test_use_segments_without_multi_segment_raises(tmp_path):
    paths, y = _make(tmp_path)
    with pytest.raises(ValueError, match="n_segments"):
        FeatureDataset(paths, y, n_segments=1, use_segments=(0,))


def test_default_is_unchanged(tmp_path):
    """不传新参数时行为必须和以前完全一致 —— 这是个改动了共用类的功能。"""
    paths, y = _make(tmp_path)
    x = FeatureDataset(paths, y)[0][0].numpy()
    assert x.shape == (N_SEG * FRAMES_PER_SEG, DIM)
