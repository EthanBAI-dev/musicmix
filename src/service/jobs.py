"""后台分离任务：上传一个音频 → 分离四轨 → 写进混音台能读的位置。

这是 P6 服务化的最小可用版本。刻意只用标准库 + 已有的分离代码，
不引 FastAPI / Celery / Redis —— 单机自用场景下，一个线程池 + 内存里的任务表就够，
引入三个新依赖只会让"零构建"的前端也变成需要 docker-compose 才能跑。

.. warning::
   **这个服务会接收任意文件并在本机上跑解码与推理，只能绑 127.0.0.1。**
   :mod:`scripts.serve` 里对非本地绑定做了显式拦截。

设计上的两个决定：

- **模型常驻**：htdemucs 首次加载要十几秒。放进模块级单例，第二首歌起就省掉了。
- **进度分阶段而不是百分比**：分离是一次不可分割的前向，中途拿不到真实百分比。
  与其编一个匀速增长的假进度条，不如如实报当前阶段（解码 / 分离 / 编码）。
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

WEB_MINE = Path("web/mine")
STEMS = ("vocals", "drums", "bass", "other")

# 上传大小上限。一首 10 分钟的无损也就 100 MB 左右；
# 设上限是为了避免手滑传个几 GB 的文件把磁盘写满。
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


@dataclass
class Job:
    id: str
    filename: str
    status: str = "queued"          # queued / running / done / error
    stage: str = "排队中"
    error: str = ""
    # 部分失败：分离成功但分析失败时用这个。与 error 分开 ——
    # 把它写进 error 会让前端把一个可用的结果显示成失败
    warning: str = ""
    track_id: str = ""
    duration: float = 0.0
    elapsed: float = 0.0
    created: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "filename": self.filename, "status": self.status,
            "stage": self.stage, "error": self.error, "warning": self.warning,
            "track_id": self.track_id,
            "duration": round(self.duration, 1), "elapsed": round(self.elapsed, 1),
        }


_JOBS: dict[str, Job] = {}
_LOCK = threading.Lock()
_SEPARATOR = None
_SEP_LOCK = threading.Lock()

# 同一时刻只跑一个分离任务。MPS 上并发前向不会更快，只会互相抢显存，
# 而且用户就一个人，排队反而让进度可预期。
_RUN_LOCK = threading.Lock()


def get_job(job_id: str) -> Job | None:
    with _LOCK:
        return _JOBS.get(job_id)


def recent_jobs(limit: int = 10) -> list[dict]:
    with _LOCK:
        js = sorted(_JOBS.values(), key=lambda j: j.created, reverse=True)
    return [j.as_dict() for j in js[:limit]]


def _separator(model: str):
    """模型常驻。首次加载十几秒，之后每首歌都省掉。"""
    global _SEPARATOR
    with _SEP_LOCK:
        if _SEPARATOR is None or _SEPARATOR[0] != model:
            from src.separation import demucs_model
            _SEPARATOR = (model, demucs_model.load(model))
        return _SEPARATOR[1]


def submit(raw: bytes, filename: str, model: str = "htdemucs",
           seconds: float = 0.0) -> Job:
    """收下上传的字节，起一个后台线程去分离。立即返回，不阻塞 HTTP。"""
    job = Job(id=uuid.uuid4().hex[:12], filename=filename)
    with _LOCK:
        _JOBS[job.id] = job
    threading.Thread(target=_run, args=(job, raw, filename, model, seconds),
                     daemon=True).start()
    return job


def _run(job: Job, raw: bytes, filename: str, model: str, seconds: float) -> None:
    import numpy as np

    from scripts.separate_file import slugify, to_mp3, update_manifest
    from src.audio.io import load_audio, save_audio
    from src.separation import demucs_model

    tmp_dir = Path("data/_uploads")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # 原样保留后缀，让 ffmpeg/soundfile 能按格式解码
    suffix = Path(filename).suffix or ".mp3"
    tmp_src = tmp_dir / f"{job.id}{suffix}"
    t0 = time.perf_counter()

    try:
        tmp_src.write_bytes(raw)

        with _RUN_LOCK:                       # 排队：一次只跑一个
            job.status, job.stage = "running", "解码中"
            mix, sr = load_audio(tmp_src, stereo=True, normalize_loudness=False)
            if seconds and seconds < mix.shape[0] / sr:
                mix = mix[: int(seconds * sr)]
            job.duration = mix.shape[0] / sr

            job.stage = f"加载 {model}"
            sep = _separator(model)

            job.stage = f"分离中（{job.duration:.0f} 秒音频）"
            stems = demucs_model.separate(sep, mix)

            job.stage = "编码 mp3"
            slug = slugify(Path(filename).stem)
            out = WEB_MINE / slug
            tmp_wav = out / "_tmp.wav"
            files = {}
            for name, y in [("mixture", mix), *stems.items()]:
                save_audio(tmp_wav, y, sr, subtype="PCM_16")
                if to_mp3(tmp_wav, out / f"{name}.mp3"):
                    files[name] = f"mine/{slug}/{name}.mp3"
            tmp_wav.unlink(missing_ok=True)

            # 音乐分析（BPM / 调性 / 和弦 / 曲式）。放在分离**之后**跑，
            # 因为它比分离快得多（233 秒的歌约 6 秒），先出 stem 能让页面早点可用。
            # 分析失败不该让整个任务失败 —— 分离结果本身仍然有价值。
            analysis = None
            try:
                job.stage = "分析节拍与和弦"
                from src.analysis.pipeline import ANALYSIS_SR, analyze_audio
                import librosa
                # load_audio 返回 **(样本, 声道)**，所以并轨要 axis=1。
                # 写成 axis=0 会把**时间轴**平均掉，得到一个 2 元素数组 ——
                # 而分析在 2 个样本上照跑不误，产出 BPM 0.0、和弦为空，**不抛任何异常**。
                # 这是实际发生过的静默失败，下面的断言就是为它加的。
                mono = mix.mean(axis=1) if mix.ndim == 2 else mix
                mono = librosa.resample(mono, orig_sr=sr, target_sr=ANALYSIS_SR)
                got = len(mono) / ANALYSIS_SR
                if abs(got - job.duration) > max(1.0, job.duration * 0.02):
                    # 并轨/重采样出错时长度会离谱地对不上。宁可在这里炸掉，
                    # 也不要产出一份"看起来像分析结果"的垃圾
                    raise ValueError(
                        f"并轨后时长 {got:.2f}s 与音频 {job.duration:.2f}s 不符，疑似轴用错")
                analysis = analyze_audio(mono, ANALYSIS_SR, source=Path(filename).stem)
                (out / "analysis.json").write_text(
                    json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception as e:                      # noqa: BLE001
                job.warning = f"分析失败（分离结果不受影响）：{type(e).__name__}: {e}"

            rms = {k: float(np.sqrt(np.mean(v**2))) for k, v in stems.items()}
            update_manifest({
                "id": slug,
                "track": Path(filename).stem,
                "tag": "mine",
                "csdr_mean": None,            # 没有真值，算不了 SDR —— 绝不填假数字
                "duration": round(job.duration, 1),
                "model": sep.name,
                "stem_rms_db": {k: round(20 * np.log10(v + 1e-12), 1) for k, v in rms.items()},
                # 分析是**估计值**，不是真值 —— 前端据此显示「模型估计」
                "analysis": f"mine/{slug}/analysis.json" if analysis else None,
                "bpm": analysis["bpm"] if analysis else None,
                "key": analysis["key"] if analysis else None,
                "files": files,
            })
            job.track_id = slug

        job.status, job.stage = "done", "完成"
    except Exception as e:
        job.status = "error"
        job.stage = "失败"
        job.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    finally:
        job.elapsed = time.perf_counter() - t0
        tmp_src.unlink(missing_ok=True)       # 上传的原文件不留存
