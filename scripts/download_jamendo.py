"""按需下载 MTG-Jamendo 的音频分块。

    python -m scripts.download_jamendo --chunks 20              # 下 00~19 块
    python -m scripts.download_jamendo --chunks 20 --type audio-low
    python -m scripts.download_jamendo --meta-only              # 只取元数据

官方的 ``scripts/download/download.py`` 只能整包下（全量 audio 有 **530 GB**），
而我们不需要全量。数据集按曲目 id 的**末两位**分成 100 个 tar 块，
实测每块 556±22 首（变异系数 4%），所以**取前 N 块 = 一份干净的随机样本**。

.. note::
   **为什么下完一块就删 tar：** 20 块的 tar 有 106 GB，解压后又是 106 GB。
   不删的话峰值要 212 GB。改成"下一块 → 校验 → 解压 → 删 tar"，
   峰值就只比最终占用多一个块（5.3 GB）。

.. warning::
   MTG-Jamendo 的音频是 Jamendo 上的 CC 授权曲目，**元数据 CC BY-NC-SA 4.0，
   仅限非商业研究与学术使用**；商业使用需 Jamendo S.A. 书面授权。
   下载下来的音频**不要入 git、不要再分发**。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

MIRRORS = {
    "mtg-fast": "https://cdn.freesound.org/mtg-jamendo",
    "mtg": "https://essentia.upf.edu/documentation/datasets/mtg-jamendo",
}
META_BASE = "https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/master/data"

# 实测每块大小（GB），用来估算总量和剩余时间
CHUNK_GB = {"audio": 5.3, "audio-low": 1.65, "melspecs": 2.3}

DEFAULT_ROOT = Path("data/jamendo")


# --------------------------------------------------------------------------------------
# 元数据
# --------------------------------------------------------------------------------------

META_FILES = [
    "autotagging.tsv",
    "autotagging_instrument.tsv",
    "autotagging_moodtheme.tsv",
    "autotagging_genre.tsv",
    "autotagging_top50tags.tsv",
]
SPLIT_SUBSETS = ["autotagging", "autotagging_instrument", "autotagging_top50tags"]


def fetch(url: str, dst: Path, quiet: bool = False) -> bool:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-sfL", "--max-time", "600", "-o", str(dst), url]
    ok = subprocess.run(cmd, check=False).returncode == 0
    if not quiet:
        print(("  ✅ " if ok else "  ❌ ") + dst.name + ("" if ok else f"  ← {url}"))
    return ok


def download_meta(root: Path, splits: tuple[int, ...] = (0,)) -> None:
    """元数据只有几 MB，全下。"""
    meta = root / "meta"
    print(f"元数据 → {meta}")
    for f in META_FILES:
        fetch(f"{META_BASE}/{f}", meta / f)

    # autotagging.tsv 只是一个**指向真实文件名的一行文本**，不是数据本身。
    # 不解开这一层，后面读到的会是 31 字节的垃圾。
    ptr = meta / "autotagging.tsv"
    if ptr.exists() and ptr.stat().st_size < 200:
        real = ptr.read_text(encoding="utf-8").strip()
        print(f"  ℹ️  autotagging.tsv 是指针，实际文件为 {real}")
        fetch(f"{META_BASE}/{real}", meta / "autotagging_real.tsv")

    for s in splits:
        for subset in SPLIT_SUBSETS:
            for part in ("train", "validation", "test"):
                name = f"{subset}-{part}.tsv"
                fetch(f"{META_BASE}/splits/split-{s}/{name}",
                      meta / "splits" / f"split-{s}" / name, quiet=True)
        print(f"  ✅ split-{s} 的 train/validation/test")


# --------------------------------------------------------------------------------------
# 音频分块
# --------------------------------------------------------------------------------------

def chunk_url(mirror: str, dtype: str, idx: int) -> str:
    return f"{MIRRORS[mirror]}/raw_30s/{dtype}/raw_30s_{dtype}-{idx:02d}.tar"


def remote_size(url: str) -> int | None:
    """服务器上的文件字节数（HEAD 的 Content-Length）。取不到返回 None。"""
    r = subprocess.run(
        ["curl", "-fsIL", "--max-time", "30", url], capture_output=True, text=True, check=False)
    if r.returncode != 0:
        return None
    for line in reversed(r.stdout.splitlines()):      # 跟随重定向后取最后一段响应头
        if line.lower().startswith("content-length:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def chunk_done(dest: Path, idx: int, expected: int | None = None) -> bool:
    """该块是否已经**完整**解压。

    只看"目录存在且非空"是不够的 —— 截断的 tar 会解出一个非空但残缺的目录，
    重跑脚本时会被当成已完成直接跳过，残缺就永远留在那儿了。
    传入 ``expected``（元数据里该块应有的曲目数）才能真正判定。
    """
    d = dest / f"{idx:02d}"
    if not (d.is_dir() and any(d.iterdir())):
        return False
    if expected is None:
        return True
    have = sum(1 for _ in d.glob("*.mp3"))
    # 留 2% 容差：元数据与实际发布的文件本来就有个位数出入
    return have >= expected * 0.98


# curl 的网络类退出码。这些代表"网线断了"而不是"这个文件有问题"，
# 遇到它们应当**等一等重试同一块**，而不是判死往下走。
# 血泪教训：一次 DNS 抖动（curl 6）让剩余 16 块在 2 分钟内全部"失败"，
# 白白浪费了一整轮。
NETWORK_ERRORS = {
    5:  "无法解析代理",
    6:  "DNS 解析失败",
    7:  "无法连接",
    28: "超时",
    35: "SSL 握手失败",
    52: "服务器无响应",
    55: "发送失败",
    56: "接收失败",
}


def download_chunk(
    mirror: str, dtype: str, idx: int, dest: Path, keep_tar: bool,
    net_retries: int = 6, backoff: float = 20.0,
) -> bool:
    tar_path = dest / f"_tmp_{dtype}-{idx:02d}.tar"
    url = chunk_url(mirror, dtype, idx)

    for attempt in range(net_retries + 1):
        # 不要用 --progress-bar：它每秒刷几十行，跑 20 块能把日志刷到几十 MB，
        # 后台运行时尤其恶心。改成静默下载 + 下完后自己报速度。
        t0 = time.perf_counter()
        # -C - 断点续传：网络中断后接着下，而不是从头来
        r = subprocess.run(
            ["curl", "-fL", "--retry", "5", "--retry-delay", "5", "-C", "-",
             "--no-progress-meter", "-o", str(tar_path), url],
            check=False,
        )
        dt = time.perf_counter() - t0
        if r.returncode == 0:
            break
        if r.returncode in NETWORK_ERRORS and attempt < net_retries:
            wait = backoff * 2**attempt          # 20s → 40 → 80 → …，最长约 10 分钟
            got = tar_path.stat().st_size / 2**20 if tar_path.exists() else 0
            print(f"  ⏳ 块 {idx:02d} 网络中断（curl {r.returncode}：{NETWORK_ERRORS[r.returncode]}），"
                  f"已下 {got:.0f} MB，{wait:.0f}s 后续传（第 {attempt+1}/{net_retries} 次）", flush=True)
            time.sleep(wait)
            continue
        print(f"  ❌ 块 {idx:02d} 下载失败（curl {r.returncode}）")
        return False

    size = tar_path.stat().st_size
    mb = size / 2**20
    print(f"  ↓ {mb:.0f} MB / {dt:.0f}s = {mb/max(dt,1e-6):.1f} MB/s", flush=True)

    # **必须先比字节数。** 血泪教训：断点续传后拿到一个被截断的 tar，
    # 而 tarfile 顺序读到断点就停下、**不报错**，于是"校验通过"、解出 191/556 个文件，
    # 脚本高高兴兴打印 ✅ —— 静默丢了 6.8% 的数据，直到训练时才发现缺文件。
    # 光验"能不能读"是不够的，必须验"完不完整"。
    expected = remote_size(url)
    if expected and size != expected:
        print(f"  ❌ 块 {idx:02d} 不完整：本地 {size:,} B，服务器 {expected:,} B"
              f"（差 {(expected-size)/2**20:.0f} MB）→ 删掉重下")
        tar_path.unlink(missing_ok=True)
        return False

    try:
        with tarfile.open(tar_path) as tf:
            n = sum(1 for m in tf.getmembers() if m.isfile())
    except Exception as e:
        print(f"  ❌ 块 {idx:02d} 归档损坏：{e}（删掉重下）")
        tar_path.unlink(missing_ok=True)
        return False

    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")

    if not keep_tar:
        tar_path.unlink(missing_ok=True)     # 见模块 docstring：不删峰值翻倍
    print(f"  ✅ 块 {idx:02d}：{n} 个文件")
    return True


def expected_per_chunk(root: Path) -> dict[int, int]:
    """从元数据统计每块应有多少首。用来判定解压是否完整。"""
    tsv = root / "meta" / "autotagging_real.tsv"
    if not tsv.exists():
        return {}
    counts: dict[int, int] = {}
    with open(tsv, encoding="utf-8") as f:
        next(f, "")
        for line in f:
            parts = line.split("\t")
            if len(parts) >= 4:
                try:
                    c = int(parts[3].split("/")[0])
                except ValueError:
                    continue
                counts[c] = counts.get(c, 0) + 1
    return counts


def main() -> int:
    p = argparse.ArgumentParser(description="按需下载 MTG-Jamendo 音频分块")
    p.add_argument("--chunks", type=int, default=20,
                   help="下载到第 N 块为止（每块约 556 首）。分块 = 曲目 id 末两位，取前 N 块即随机样本")
    p.add_argument("--start", type=int, default=0,
                   help="从第几块开始（默认 0）。配合 --chunks 取区间 [start, chunks)。"
                        "用于 Colab 那种「下一块→提特征→删音频」的流式处理："
                        "全量音频 490 GB，塞不进 Colab 本地盘，必须分批")
    p.add_argument("--type", default="audio", choices=("audio", "audio-low", "melspecs"),
                   help="audio=320k 立体声；audio-low=100k **单声道**（不适合做分离）")
    p.add_argument("--mirror", default="mtg-fast", choices=tuple(MIRRORS))
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--keep-tar", action="store_true", help="保留 tar（磁盘占用翻倍）")
    p.add_argument("--retries", type=int, default=3,
                   help="失败的块自动重试几轮（半成品 tar 保留，续传不重下）")
    p.add_argument("--meta-only", action="store_true")
    args = p.parse_args()

    root = Path(args.root)
    download_meta(root)
    if args.meta_only:
        return 0

    if args.type == "audio-low":
        print("\n⚠️  audio-low 是**单声道** 100kbps。跑 demucs 分离会明显掉质量，"
              "会给 stem-aware 实验引入混淆变量。确认这是你要的。")

    dest = root / args.type
    dest.mkdir(parents=True, exist_ok=True)
    expected_counts = expected_per_chunk(root)
    if not 0 <= args.start < args.chunks:
        print(f"❌ 区间非法：--start {args.start} 必须在 [0, --chunks {args.chunks}) 内",
              file=sys.stderr)
        return 1
    todo = [i for i in range(args.start, args.chunks)
            if not chunk_done(dest, i, expected_counts.get(i))]
    done = (args.chunks - args.start) - len(todo)

    est = len(todo) * CHUNK_GB[args.type]
    print(f"\n{args.type}：区间 [{args.start}, {args.chunks}) 共 {args.chunks - args.start} 块，"
          f"已有 {done} 块，待下 {len(todo)} 块（约 {est:.0f} GB）")
    free = shutil.disk_usage(root.parent if root.exists() else ".").free / 2**30
    print(f"磁盘可用 {free:.0f} GB")
    if free < est * 1.15:
        print("❌ 磁盘空间不足（需要预留约 15% 余量给解压过程）", file=sys.stderr)
        return 1
    if not todo:
        print("✅ 全部已就绪")
        return 0

    t0 = time.perf_counter()
    failed = []
    for k, idx in enumerate(todo, 1):
        el = time.perf_counter() - t0
        eta = el / max(k - 1, 1) * (len(todo) - k + 1) if k > 1 else 0
        print(f"\n[{k}/{len(todo)}] 块 {idx:02d}" + (f"   已用 {el/60:.0f}min，剩余约 {eta/60:.0f}min" if k > 1 else ""))
        if not download_chunk(args.mirror, args.type, idx, dest, args.keep_tar):
            failed.append(idx)

    # 失败的块自动重试。网络抖动导致单块中断是常态（实测 20 块里就断了一块），
    # 每次都要人工重跑脚本很蠢。半成品 tar 保留着，-C - 会接着下而不是从头来。
    for attempt in range(1, args.retries + 1):
        if not failed:
            break
        print(f"\n{'=' * 60}\n第 {attempt}/{args.retries} 轮重试：{failed}")
        retry, failed = failed, []
        for idx in retry:
            if chunk_done(dest, idx, expected_counts.get(idx)):
                continue
            print(f"\n[重试] 块 {idx:02d}")
            if not download_chunk(args.mirror, args.type, idx, dest, args.keep_tar):
                failed.append(idx)

    total = sum(1 for d in dest.iterdir() if d.is_dir() and len(d.name) == 2 for _ in d.iterdir())
    size = sum(f.stat().st_size for f in dest.rglob("*.mp3")) / 2**30
    print(f"\n{'=' * 60}\n完成：{total} 个音频文件，{size:.0f} GB，耗时 {(time.perf_counter()-t0)/60:.0f} min")
    if failed:
        print(f"⚠️  重试 {args.retries} 轮后仍有 {len(failed)} 块失败：{failed}")
        print("   半成品 tar 已保留，重跑本脚本会从断点继续。")
        return 1
    print("下一步：python -m scripts.jamendo_stats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
