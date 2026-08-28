"""环境自检。M0 的第一道关卡。

    python -m scripts.check_env

重点检查两件容易在后面咬人的事：

1. **MPS 是否真的能算**，而不只是 ``is_available() == True``。
   PyTorch 在 Apple Silicon 上有些算子不支持会**静默回退 CPU**，
   只看可用性标志会让后面所有 RTF 数字都是假的。
2. **ffmpeg 是否在 PATH 里**。mp3/m4a 解码全靠它，缺了会在跑到一半时才炸。
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
import time

REQUIRED = [
    ("numpy", "数值计算"),
    ("scipy", "信号处理"),
    ("soundfile", "wav/flac 读写"),
    ("librosa", "MIR 特征、重采样"),
    ("pyloudnorm", "响度归一化"),
    ("sklearn", "标签评测指标", "scikit-learn"),
    ("museval", "分离评测官方口径"),
    ("musdb", "MUSDB18 数据接口"),
    ("mir_eval", "节奏/和弦评测"),
    ("pytest", "单元测试"),
]
OPTIONAL = [
    ("torch", "模型推理与训练"),
    ("torchaudio", "音频张量算子"),
    ("pandas", "结果表"),
    ("matplotlib", "画图"),
]


def _check_imports(items, required: bool) -> list[str]:
    problems = []
    for item in items:
        mod, desc = item[0], item[1]
        dist = item[2] if len(item) > 2 else mod
        try:
            __import__(mod)
            try:
                ver = md.version(dist)
            except Exception:
                ver = "?"
            print(f"  ✅ {mod:<14} {ver:<12} {desc}")
        except ImportError:
            mark = "❌" if required else "⚠️ "
            print(f"  {mark} {mod:<14} {'缺失':<12} {desc}")
            if required:
                problems.append(f"缺少必需依赖 {dist}")
    return problems


def _check_torch_device() -> list[str]:
    problems: list[str] = []
    try:
        import torch
    except ImportError:
        print("  ⚠️  未安装 torch，跳过设备检查")
        return problems

    print(f"  torch {torch.__version__}")
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"  CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
    print(f"  设备: {device}")

    if device == "cpu":
        print("  ⚠️  没有可用加速器，P2/P4 的训练会非常慢（可以用 Colab）")
        return problems

    # 真的跑一次矩阵乘，并和 CPU 结果对比 —— 只看 is_available() 是不够的
    try:
        a = torch.randn(1024, 1024, device=device)
        b = torch.randn(1024, 1024, device=device)
        _ = a @ b
        if device == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(10):
            c = a @ b
        if device == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gflops = 10 * 2 * 1024**3 / dt / 1e9
        print(f"  ✅ {device} 实测可用：1024³ matmul ×10 = {dt * 1000:.0f} ms（{gflops:.0f} GFLOPS）")

        err = (c.cpu() - (a.cpu() @ b.cpu())).abs().max().item()
        if err > 1e-2:
            problems.append(f"{device} 计算结果与 CPU 偏差过大（{err:.3g}）")
        else:
            print(f"  ✅ 与 CPU 结果一致（最大偏差 {err:.2e}）")
    except Exception as e:
        problems.append(f"{device} 实测失败：{e}")
    return problems


def _check_ffmpeg() -> list[str]:
    exe = shutil.which("ffmpeg")
    if not exe:
        print("  ❌ ffmpeg 不在 PATH 里 —— mp3/m4a 解码会失败")
        print("     macOS: brew install ffmpeg")
        return ["缺少 ffmpeg"]
    ver = subprocess.run([exe, "-version"], capture_output=True, text=True, check=False)
    print(f"  ✅ {ver.stdout.splitlines()[0][:60]}")
    return []


def main() -> int:
    print("=" * 70)
    print("环境自检")
    print("=" * 70)

    print(f"\n[系统]\n  {platform.platform()}")
    print(f"  Python {platform.python_version()} @ {sys.executable}")

    print("\n[必需依赖]")
    problems = _check_imports(REQUIRED, required=True)

    print("\n[可选依赖]")
    _check_imports(OPTIONAL, required=False)

    print("\n[加速器]")
    problems += _check_torch_device()

    print("\n[外部工具]")
    problems += _check_ffmpeg()

    print("\n" + "=" * 70)
    if problems:
        print(f"❌ {len(problems)} 个问题：")
        for p in problems:
            print(f"   - {p}")
        return 1
    print("✅ 环境就绪。下一步：")
    print("   python -m pytest tests/ -q")
    print("   python -m scripts.run_separation_eval --synthetic 8 --model oracle")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
