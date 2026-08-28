"""性能测量工具的单测。

计时代码测起来天生有点绕（不能断言绝对耗时，机器负载会让测试随机失败），
所以这里只断言**结构性质**：预热被丢弃、RTF 算法正确、单位换算正确。
"""

import time

import pytest

from src.bench.timer import (
    BenchResult,
    benchmark,
    device_info,
    format_bench_table,
    peak_rss_bytes,
    timer,
)


def test_timer_measures_elapsed():
    with timer("sleep") as t:
        time.sleep(0.05)
    assert t.elapsed >= 0.045


def test_timer_reports_zero_before_exit():
    with timer() as t:
        assert t.elapsed == 0.0
    assert t.elapsed > 0.0


def test_benchmark_runs_expected_number_of_times():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1

    r = benchmark(fn, "counter", n_runs=5, n_warmup=2)
    assert calls["n"] == 7          # 2 次预热 + 5 次正式
    assert r.n_runs == 5


def test_benchmark_discards_warmup_from_statistics():
    """第一次刻意很慢，之后很快。中位数应当反映稳态，而不是被预热拖垮。

    这条防的是"忘记预热导致 RTF 虚高几倍"这个最常见的性能测量错误。
    """
    state = {"first": True}

    def fn():
        if state["first"]:
            state["first"] = False
            time.sleep(0.15)

    r = benchmark(fn, "warmup-heavy", n_runs=5, n_warmup=1)
    assert r.median_s < 0.01, f"预热没被丢掉，中位数 {r.median_s:.4f}s"


def test_rtf_and_speedup():
    r = BenchResult(
        label="sep", median_s=6.0, mean_s=6.0, std_s=0.0, min_s=6.0,
        n_runs=5, audio_seconds=180.0,
    )
    assert r.rtf == pytest.approx(6.0 / 180.0)
    assert r.speedup == pytest.approx(30.0)


def test_rtf_is_none_without_audio_duration():
    r = BenchResult(label="x", median_s=1.0, mean_s=1.0, std_s=0.0, min_s=1.0, n_runs=1)
    assert r.rtf is None
    assert r.speedup is None


def test_benchmark_computes_rtf_when_given_duration():
    r = benchmark(lambda: time.sleep(0.02), "fake", n_runs=3, n_warmup=1, audio_seconds=10.0)
    assert r.rtf is not None
    assert r.rtf < 0.01


def test_peak_rss_is_plausible():
    """跨平台单位归一化：Python 进程的 RSS 至少几十 MB，不可能是几十 KB。

    这条防的正是 ru_maxrss 在 macOS(字节) 和 Linux(KB) 上单位不同的坑 ——
    不处理的话本机和 Colab 上测出来的数字会差 1024 倍。
    """
    rss = peak_rss_bytes()
    assert 10e6 < rss < 100e9, f"峰值 RSS = {rss} 字节，看起来单位换算错了"


def test_describe_includes_rtf():
    r = BenchResult(
        label="sep", median_s=6.0, mean_s=6.1, std_s=0.2, min_s=5.9,
        n_runs=5, audio_seconds=180.0, peak_rss_mb=1234.0,
    )
    text = r.describe()
    assert "RTF=" in text
    assert "×30.0" in text


def test_format_bench_table():
    rows = [
        BenchResult("解码", 0.5, 0.5, 0.0, 0.5, 5, audio_seconds=180.0, peak_rss_mb=100.0),
        BenchResult("分离", 20.0, 20.0, 0.0, 20.0, 5, audio_seconds=180.0, peak_rss_mb=4000.0),
    ]
    table = format_bench_table(rows)
    assert "| 解码 |" in table
    assert "| 分离 |" in table
    assert "RTF" in table


def test_device_info_reports_device():
    info = device_info()
    assert "device" in info
    assert "platform" in info
