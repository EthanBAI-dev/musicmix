"""推理时间与内存测量。

要拿得出手的性能数字，规矩只有三条：

1. **必须预热。** 第一次调用包含权重加载、算子编译、显存分配，比稳态慢几倍。
   一律丢弃第一次。
2. **报中位数，不报单次。** 后台进程会污染任何一次单独的测量。
3. **MPS 上必须同步再停表。** PyTorch 的 MPS/CUDA 是异步下发的，
   不 synchronize 就停表测到的是"下发耗时"，不是"计算耗时"，
   数字会好看得离谱且完全是假的。
"""

from __future__ import annotations

import gc
import platform
import resource
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


def _sync() -> None:
    """等待加速器上的异步任务完成。没装 torch 时静默跳过。"""
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def peak_rss_bytes() -> int:
    """当前进程的峰值常驻内存。

    ``ru_maxrss`` 的单位**跨平台不一致**：macOS 是字节，Linux 是 KB。
    这里按平台归一到字节 —— 这个坑不处理的话，Mac 上和 Colab 上测出来的数字会差 1024 倍。
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(rss) if platform.system() == "Darwin" else int(rss) * 1024


def accelerator_memory_bytes() -> int | None:
    """当前加速器已分配显存；不可用时返回 None。"""
    try:
        import torch
    except ImportError:
        return None
    if torch.backends.mps.is_available():
        return int(torch.mps.current_allocated_memory())
    if torch.cuda.is_available():
        return int(torch.cuda.max_memory_allocated())
    return None


@contextmanager
def timer(label: str = "", verbose: bool = False):
    """计时上下文管理器，退出时自动 synchronize。

    >>> with timer("stft") as t:
    ...     y = do_stft(x)
    >>> t.elapsed
    """

    class _T:
        elapsed: float = 0.0

    t = _T()
    _sync()
    gc.collect()
    start = time.perf_counter()
    try:
        yield t
    finally:
        _sync()
        t.elapsed = time.perf_counter() - start
        if verbose:
            print(f"[timer] {label}: {t.elapsed:.4f} s")


@dataclass
class BenchResult:
    """一次基准测量的结果。"""

    label: str
    median_s: float
    mean_s: float
    std_s: float
    min_s: float
    n_runs: int
    audio_seconds: float | None = None
    peak_rss_mb: float | None = None
    accel_mem_mb: float | None = None

    @property
    def rtf(self) -> float | None:
        """实时率 = 处理耗时 / 音频时长。RTF < 1 表示比实时快。"""
        if not self.audio_seconds:
            return None
        return self.median_s / self.audio_seconds

    @property
    def speedup(self) -> float | None:
        """比实时快多少倍。RTF 的倒数，汇报时比 RTF 直观。"""
        r = self.rtf
        return None if not r else 1.0 / r

    def describe(self) -> str:
        parts = [f"{self.label}: {self.median_s:.3f}s (中位数, n={self.n_runs})"]
        if self.rtf is not None:
            parts.append(f"RTF={self.rtf:.4f} (×{self.speedup:.1f} 实时)")
        if self.peak_rss_mb is not None:
            parts.append(f"峰值RSS={self.peak_rss_mb:.0f}MB")
        if self.accel_mem_mb:
            parts.append(f"加速器内存={self.accel_mem_mb:.0f}MB")
        return " | ".join(parts)


def benchmark(
    fn: Callable[[], Any],
    label: str = "fn",
    n_runs: int = 5,
    n_warmup: int = 1,
    audio_seconds: float | None = None,
) -> BenchResult:
    """跑 ``n_warmup + n_runs`` 次，丢掉预热，返回统计量。

    Args:
        fn: 无参可调用对象。有参数就用 ``functools.partial`` 包一层。
        audio_seconds: 传了才会算 RTF。
    """
    import statistics

    for _ in range(n_warmup):
        fn()
    _sync()

    times: list[float] = []
    for _ in range(n_runs):
        with timer() as t:
            fn()
        times.append(t.elapsed)

    accel = accelerator_memory_bytes()
    return BenchResult(
        label=label,
        median_s=statistics.median(times),
        mean_s=statistics.fmean(times),
        std_s=statistics.pstdev(times) if len(times) > 1 else 0.0,
        min_s=min(times),
        n_runs=n_runs,
        audio_seconds=audio_seconds,
        peak_rss_mb=peak_rss_bytes() / 1e6,
        accel_mem_mb=accel / 1e6 if accel else None,
    )


def format_bench_table(results: list[BenchResult]) -> str:
    """渲染成 markdown，直接贴进 ``results/latency.md``。"""
    lines = [
        "| 阶段 | 中位耗时(s) | 均值±标准差 | RTF | 比实时快 | 峰值RSS(MB) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        rtf = f"{r.rtf:.4f}" if r.rtf is not None else "—"
        sp = f"×{r.speedup:.1f}" if r.speedup is not None else "—"
        rss = f"{r.peak_rss_mb:.0f}" if r.peak_rss_mb is not None else "—"
        lines.append(
            f"| {r.label} | {r.median_s:.3f} | {r.mean_s:.3f}±{r.std_s:.3f} | {rtf} | {sp} | {rss} |"
        )
    return "\n".join(lines)


def device_info() -> dict[str, str]:
    """记进每张结果表表头的硬件信息。没有它，三个月后没人知道数字是在哪测的。"""
    import subprocess

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        try:
            info["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=False,
            ).stdout.strip()
        except Exception:
            pass
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.backends.mps.is_available():
            info["device"] = "mps"
        elif torch.cuda.is_available():
            info["device"] = f"cuda ({torch.cuda.get_device_name(0)})"
        else:
            info["device"] = "cpu"
    except ImportError:
        info["device"] = "cpu (no torch)"
    return info
