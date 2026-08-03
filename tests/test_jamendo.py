"""MTG-Jamendo 数据访问的单测。

不依赖真实数据集 —— 用 tmp_path 造一份最小的 tsv + 目录结构。
重点钉住三件容易出错、且出错时**不报错只是结果变差**的事：
1. `autotagging.tsv` 是指针文件，误读会得到垃圾
2. 本地只有子样本时，split 必须按已下载的块过滤
3. 词表由训练集决定、并能按最少正样本数裁剪
"""

import numpy as np
import pytest

from src.datasets.jamendo import (
    TagVocab,
    Track,
    available_chunks,
    load_split,
    load_subset,
    tag_statistics,
)

HEADER = "TRACK_ID\tARTIST_ID\tALBUM_ID\tPATH\tDURATION\tTAGS\n"


def _row(tid: int, chunk: int, tags: list[str], dur: float = 120.0) -> str:
    return (f"track_{tid:07d}\tartist_1\talbum_1\t{chunk:02d}/{tid}.mp3\t{dur}\t"
            + "\t".join(tags) + "\n")


@pytest.fixture
def root(tmp_path):
    """造一个迷你数据集：块 00 和 01 有音频，块 02 没有。"""
    meta = tmp_path / "meta" / "splits" / "split-0"
    meta.mkdir(parents=True)

    rows = [
        _row(100, 0, ["genre---rock", "instrument---guitar"]),
        _row(200, 0, ["genre---rock", "mood/theme---happy"]),
        _row(101, 1, ["genre---jazz", "instrument---piano"]),
        _row(201, 1, ["genre---rock"]),
        _row(102, 2, ["genre---metal", "instrument---guitar"]),   # 块 02 无音频
    ]
    (tmp_path / "meta" / "autotagging_real.tsv").write_text(HEADER + "".join(rows), encoding="utf-8")

    # 指针文件：真实数据集里 autotagging.tsv 就是这么一行
    (tmp_path / "meta" / "autotagging.tsv").write_text(
        "raw_30s_cleantags_50artists.tsv\n", encoding="utf-8")

    (meta / "autotagging-train.tsv").write_text(HEADER + rows[0] + rows[2] + rows[4], encoding="utf-8")
    (meta / "autotagging-validation.tsv").write_text(HEADER + rows[1], encoding="utf-8")
    (meta / "autotagging-test.tsv").write_text(HEADER + rows[3], encoding="utf-8")

    for c in (0, 1):
        d = tmp_path / "audio" / f"{c:02d}"
        d.mkdir(parents=True)
        (d / "dummy.mp3").write_bytes(b"\x00")
    return tmp_path


# --------------------------------------------------------------------------------------
# 指针文件
# --------------------------------------------------------------------------------------

def test_reading_the_pointer_file_raises_a_clear_error(root):
    """**误读 autotagging.tsv 必须报错，而不是静默返回空列表。**

    真实数据集里它只有 31 字节、内容是另一个文件名。
    如果这里静默返回空，后面训练会在"0 条样本"上跑得很开心。
    """
    with pytest.raises(ValueError, match="指针文件|不是标注文件"):
        load_subset("autotagging", root=root, only_local=False)


def test_reading_the_real_file_works(root):
    tracks = load_subset("autotagging_real", root=root, only_local=False)
    assert len(tracks) == 5
    assert tracks[0].track_id == "track_0000100"
    assert tracks[0].tags == ("genre---rock", "instrument---guitar")


# --------------------------------------------------------------------------------------
# 只保留本地已下载的块
# --------------------------------------------------------------------------------------

def test_available_chunks_finds_only_nonempty_dirs(root):
    (root / "audio" / "03").mkdir()          # 空目录不算
    assert available_chunks(root) == {0, 1}


def test_load_subset_filters_to_local_chunks(root):
    """块 02 没下载 → 它的曲目必须被剔除，否则训练时会读到不存在的文件。"""
    tracks = load_subset("autotagging_real", root=root, only_local=True)
    assert {t.chunk for t in tracks} == {0, 1}
    assert len(tracks) == 4


def test_missing_audio_gives_actionable_error(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "autotagging_real.tsv").write_text(HEADER + _row(1, 0, ["genre---rock"]), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="download_jamendo"):
        load_subset("autotagging_real", root=tmp_path, only_local=True)


def test_load_split_filters_every_part(root):
    parts, _ = load_split("autotagging", root=root, only_local=True)
    assert len(parts["train"]) == 2          # 块 02 的那条被剔除
    assert len(parts["validation"]) == 1
    assert len(parts["test"]) == 1


# --------------------------------------------------------------------------------------
# 词表
# --------------------------------------------------------------------------------------

def test_vocab_comes_from_train_only(root):
    """词表必须由训练集决定 —— 用上验证/测试集的标签就是信息泄漏。"""
    parts, vocab = load_split("autotagging", root=root, only_local=True)
    # 训练集（过滤后）只有 track_100 和 track_101
    assert set(vocab.tags) == {"genre---rock", "instrument---guitar",
                               "genre---jazz", "instrument---piano"}
    assert "mood/theme---happy" not in vocab.index      # 只在验证集出现


def test_min_positives_prunes_rare_tags(root):
    parts, vocab = load_split("autotagging", root=root, only_local=False, min_positives=2)
    # 全部三条训练样本里，只有 guitar 出现了 2 次
    assert vocab.tags == ("instrument---guitar",)


def test_vocab_groups_split_by_category():
    vocab = TagVocab(("genre---rock", "genre---jazz", "instrument---guitar", "mood/theme---sad"))
    g = vocab.groups
    assert set(g) == {"genre", "instrument", "mood/theme"}
    assert len(g["genre"]) == 2
    assert len(g["instrument"]) == 1


def test_encode_produces_multihot_matrix():
    vocab = TagVocab(("genre---rock", "instrument---guitar"))
    tracks = [
        Track("a", "x", "y", "00/1.mp3", 10.0, ("genre---rock",)),
        Track("b", "x", "y", "00/2.mp3", 10.0, ("genre---rock", "instrument---guitar")),
        Track("c", "x", "y", "00/3.mp3", 10.0, ("genre---unknown",)),   # 不在词表 → 全 0
    ]
    y = vocab.encode(tracks)
    assert y.shape == (3, 2)
    assert y.dtype == np.float32
    assert y[0].tolist() == [1.0, 0.0]
    assert y[1].tolist() == [1.0, 1.0]
    assert y[2].tolist() == [0.0, 0.0]


def test_audio_path_handles_low_variant():
    t = Track("a", "x", "y", "07/12345.mp3", 10.0, ())
    assert str(t.audio_path(root=__import__("pathlib").Path("R"))).endswith("R/audio/07/12345.mp3")
    assert str(t.audio_path(root=__import__("pathlib").Path("R"), dtype="audio-low")).endswith(
        "R/audio-low/07/12345.low.mp3")


# --------------------------------------------------------------------------------------
# 统计
# --------------------------------------------------------------------------------------

def test_tag_statistics_basic(root):
    tracks = load_subset("autotagging_real", root=root, only_local=False)
    st = tag_statistics(tracks)
    assert st["n_tracks"] == 5
    assert st["n_tags"] == 6
    assert st["freq_max"] == 3           # genre---rock 出现 3 次
    assert st["tags_per_track"] == pytest.approx(1.8)
    assert set(st["by_category"]) == {"genre", "instrument", "mood/theme"}


def test_tag_statistics_respects_vocab(root):
    tracks = load_subset("autotagging_real", root=root, only_local=False)
    st = tag_statistics(tracks, TagVocab(("genre---rock",)))
    assert st["n_tags"] == 1
    assert st["counts"] == {"genre---rock": 3}
