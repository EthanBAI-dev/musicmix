"""MUSDB18-HQ 数据访问。

MUSDB18-HQ 是**未压缩 WAV 版**（不是 stem.mp4 版），目录结构：

```
data/musdb18hq/
├── train/  (100 首)
│   └── <曲名>/{mixture,vocals,drums,bass,other}.wav
└── test/   (50 首)
    └── <曲名>/{mixture,vocals,drums,bass,other}.wav
```

下载见 :func:`download_hint`。约 30 GB，**不要入 git**。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio.io import load_audio

STEMS = ("vocals", "drums", "bass", "other")
MIXTURE = "mixture"

DEFAULT_ROOT = Path("data/musdb18hq")

# 从 100 首训练集里固定切出的验证集。**种子性质的常量，永远不要改。**
# 沿用 Demucs 官方仓库的划分，这样验证集数字能和社区结果对上。
VALIDATION_TRACKS = (
    "A Classic Education - NightOwl",
    "ANiMAL - Clinic A",
    "Actions - Devil's Words",
    "Alexander Ross - Velvet Curtain",
    "Aimee Norwich - Child",
    "Ben Carrigan - We'll Talk About It All Tonight",
    "Bill Chudziak - Children Of No-one",
    "Clara Berry And Wooldog - Air Traffic",
    "Fergessen - Back From The Start",
    "James May - On The Line",
    "Leaf - Summerghost",
    "Meaxic - You Listen",
    "Patrick Talbot - A Reason To Leave",
    "Triviul - Angelsaint",
)


def download_hint(root: Path = DEFAULT_ROOT) -> str:
    return (
        f"未在 {root.resolve()} 找到 MUSDB18-HQ。\n\n"
        "获取方式（约 30 GB，需要在 Zenodo 上同意非商业研究用途的条款）：\n"
        "  1. 打开 https://zenodo.org/records/3338373 申请并下载 musdb18hq.zip\n"
        f"  2. 解压到 {root}/，确认里面有 train/ 和 test/ 两个子目录\n"
        f"  3. 跑 `python -m scripts.check_data` 校验完整性\n\n"
        "注意：要的是 **-HQ 版**（未压缩 WAV），不是原版 musdb18（stem.mp4）。"
    )


@dataclass
class Track:
    """一首歌。音频是**懒加载**的 —— MUSDB18-HQ 全量 30GB，不可能一次读进内存。"""

    name: str
    path: Path
    subset: str

    def audio(self, stem: str) -> np.ndarray:
        """读一个声部，返回 ``(n, 2) float32 @ 44.1kHz``。

        ``stem`` ∈ ``{"mixture", "vocals", "drums", "bass", "other"}``
        """
        f = self.path / f"{stem}.wav"
        if not f.exists():
            raise FileNotFoundError(f"{self.name} 缺少 {stem}.wav")
        # 评测路径：绝不做响度归一化，SDR 是幅度敏感的
        x, _ = load_audio(f, stereo=True, normalize_loudness=False)
        return x

    def references(self) -> dict[str, np.ndarray]:
        """四轨真值 ``{stem: (n, 2)}``。"""
        return {s: self.audio(s) for s in STEMS}

    def mixture(self) -> np.ndarray:
        return self.audio(MIXTURE)

    @property
    def duration_seconds(self) -> float:
        import soundfile as sf

        info = sf.info(str(self.path / f"{MIXTURE}.wav"))
        return float(info.frames) / info.samplerate


def load_tracks(
    root: Path | str = DEFAULT_ROOT,
    subset: str = "test",
    exclude_validation: bool = False,
    only_validation: bool = False,
) -> list[Track]:
    """列出某个子集里的所有歌。

    Args:
        subset: ``"train"`` 或 ``"test"``。
        exclude_validation: 只在 ``subset="train"`` 时有意义，
            剔除 :data:`VALIDATION_TRACKS`，得到 86 首真正的训练歌。
        only_validation: 反过来，只要那 14 首验证歌。

    Raises:
        FileNotFoundError: 数据不存在时抛出，附带下载指引。
    """
    root = Path(root)
    sub = root / subset
    if not sub.is_dir():
        raise FileNotFoundError(download_hint(root))

    tracks = [
        Track(name=d.name, path=d, subset=subset)
        for d in sorted(sub.iterdir())
        if d.is_dir() and (d / f"{MIXTURE}.wav").exists()
    ]
    if not tracks:
        raise FileNotFoundError(f"{sub} 下没有找到任何完整曲目。\n\n{download_hint(root)}")

    val = set(VALIDATION_TRACKS)
    if only_validation:
        tracks = [t for t in tracks if t.name in val]
    elif exclude_validation:
        tracks = [t for t in tracks if t.name not in val]
    return tracks


def verify(root: Path | str = DEFAULT_ROOT) -> dict[str, object]:
    """完整性校验：曲目数、缺失文件、采样率与声道数是否符合预期。

    M0 的验收项之一。返回一个报告字典，同时把问题打印出来。
    """
    import soundfile as sf

    root = Path(root)
    report: dict[str, object] = {"root": str(root.resolve()), "ok": True, "problems": []}
    problems: list[str] = report["problems"]  # type: ignore[assignment]

    expected = {"train": 100, "test": 50}
    for subset, n_expected in expected.items():
        try:
            tracks = load_tracks(root, subset)
        except FileNotFoundError as e:
            problems.append(str(e).splitlines()[0])
            report["ok"] = False
            continue

        report[f"{subset}_count"] = len(tracks)
        if len(tracks) != n_expected:
            problems.append(f"{subset}: 期望 {n_expected} 首，实际 {len(tracks)} 首")
            report["ok"] = False

        # 逐首查文件齐全 + 规格；只读文件头，不解码，很快
        for t in tracks:
            for stem in (MIXTURE, *STEMS):
                f = t.path / f"{stem}.wav"
                if not f.exists():
                    problems.append(f"{subset}/{t.name}: 缺少 {stem}.wav")
                    report["ok"] = False
                    continue
                info = sf.info(str(f))
                if info.samplerate != 44100:
                    problems.append(f"{subset}/{t.name}/{stem}: 采样率 {info.samplerate} ≠ 44100")
                    report["ok"] = False
                if info.channels != 2:
                    problems.append(f"{subset}/{t.name}/{stem}: {info.channels} 声道 ≠ 2")
                    report["ok"] = False

    if report["ok"]:
        print(f"✅ MUSDB18-HQ 校验通过：train={report.get('train_count')} test={report.get('test_count')}")
    else:
        print(f"❌ 发现 {len(problems)} 个问题：")
        for p in problems[:20]:
            print(f"   - {p}")
        if len(problems) > 20:
            print(f"   ... 还有 {len(problems) - 20} 个")
    return report
