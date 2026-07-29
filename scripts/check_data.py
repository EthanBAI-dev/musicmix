"""数据集完整性校验。

    python -m scripts.check_data                    # 检查 MUSDB18-HQ
    python -m scripts.check_data --root /path/to/musdb18hq

30GB 的数据集下载中断很常见，而且**损坏的表现往往不是报错，是数字悄悄变得难看**。
所以在跑任何评测之前先过一遍这个脚本。
"""

from __future__ import annotations

import argparse
import sys

from src.datasets.musdb import DEFAULT_ROOT, download_hint, verify


def main() -> int:
    p = argparse.ArgumentParser(description="校验 MUSDB18-HQ 完整性")
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    args = p.parse_args()

    print("=" * 70)
    print(f"校验 MUSDB18-HQ：{args.root}")
    print("=" * 70)

    report = verify(args.root)
    if not report["ok"]:
        problems = report["problems"]
        if problems and "未在" in str(problems[0]):
            print()
            print(download_hint(args.root))
        return 1

    print("\n下一步：")
    print("  python -m scripts.run_separation_eval --model trivial --subset test --out results/p1_trivial.md")
    print("  python -m scripts.run_separation_eval --model oracle  --subset test --out results/p1_oracle.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
