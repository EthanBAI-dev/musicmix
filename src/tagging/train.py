"""标签模型的训练与评测循环。

设计上只做一件事：**让 L0→L4 这条阶梯的每一级都只改变一个因素**，
这样每一级相对上一级的提升才能干净地归因。

::

    L0  mel + 小 CNN 从头训
    L1  冻结基座 + 线性探针（平均池化）
    L2  L1 + 注意力池化 + MLP 头       ← 只变了池化与头部
    L3  L2 + 不平衡损失（ASL/Focal）    ← 只变了损失
    L4  L3 + 逐标签阈值优化            ← 只变了推理期的阈值，模型完全不动

**L4 尤其要注意**：它不重新训练任何东西，只是在验证集上给每个标签单独搜一个阈值。
所以它的 mAP 必然与 L3 **一字不差**（mAP 与阈值无关），只有 F1 会变。
这个"mAP 不动、Macro-F1 大涨"的对比本身就是结论。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.eval.tagging import (
    DEFAULT_THRESHOLD_GRID,
    TaggingScores,
    evaluate_tagging,
    tune_thresholds,
)


@dataclass
class TrainConfig:
    level: str = "L0"
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "bce"
    pooling: str = "mean"
    patience: int = 6
    device: str = "auto"
    seed: int = 0
    # 训练时随机裁剪的帧数；None 表示用整段。随机裁剪同时起到数据增强的作用
    crop_frames: int | None = 512
    num_workers: int = 4


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    val_map: float
    val_macro_f1: float
    seconds: float


@dataclass
class TrainResult:
    level: str
    config: dict
    best_epoch: int
    history: list[EpochLog] = field(default_factory=list)
    val_scores: TaggingScores | None = None
    test_scores: TaggingScores | None = None
    thresholds: np.ndarray | None = None
    n_params: int = 0
    train_seconds: float = 0.0
    # 训练好的权重（早停选出的最优 epoch）。检索与部署都要用它，
    # 只留在内存里意味着每次都得重训 —— 96 秒虽不贵，但结果会因种子而异，
    # 拿"另一次训练"的模型去解释"这一次训练"的指标是不对的。
    state_dict: dict | None = None


# --------------------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------------------

@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(y_true, y_score)``，都是 ``(n_samples, n_tags)`` 的 numpy。"""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        logits = model(x.to(device, non_blocking=True))
        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(ps)


# --------------------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------------------

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainConfig,
    tag_groups: dict[str, np.ndarray] | None = None,
    test_loader: DataLoader | None = None,
    log_every: int = 1,
) -> TrainResult:
    """训练到验证 mAP 不再提升为止，然后（可选）在测试集上评一次。

    Note:
        **早停看的是 mAP 而不是 Macro-F1。**
        因为 F1 依赖阈值，而训练期间用的是固定 0.5 —— 拿一个受阈值污染的指标
        做早停，会让模型选择偏向"恰好在 0.5 附近好用"的解。
        mAP 只看排序，是训练期唯一干净的模型选择信号。
        阈值留到 L4 单独处理。
    """
    from src.tagging.losses import build_loss
    from src.tagging.models import count_params

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device(cfg.device)
    model = model.to(device)
    criterion = build_loss(cfg.loss)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)

    result = TrainResult(level=cfg.level, config=asdict(cfg), best_epoch=-1,
                         n_params=count_params(model))
    best_map, best_state, bad = -1.0, None, 0
    t_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        total, n_batch = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
            n_batch += 1
        sched.step()

        y_true, y_score = predict(model, val_loader, device)
        s = evaluate_tagging(y_true, y_score)
        log = EpochLog(epoch, total / max(n_batch, 1), s.map, s.macro_f1_default,
                       time.perf_counter() - t0)
        result.history.append(log)

        if epoch % log_every == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{cfg.epochs}  loss={log.train_loss:.4f}  "
                  f"val mAP={s.map:.4f}  MacroF1@0.5={s.macro_f1_default:.4f}  "
                  f"{log.seconds:.0f}s", flush=True)

        if s.map > best_map:
            best_map, result.best_epoch, bad = s.map, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                print(f"  早停于 epoch {epoch}（验证 mAP 已 {cfg.patience} 轮没有提升）")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.state_dict = best_state
    result.train_seconds = time.perf_counter() - t_start

    # ---- 阈值只在验证集上搜，然后固定住 ----
    y_val, p_val = predict(model, val_loader, device)
    result.thresholds = tune_thresholds(y_val, p_val, DEFAULT_THRESHOLD_GRID)
    result.val_scores = evaluate_tagging(y_val, p_val, result.thresholds, tag_groups)

    if test_loader is not None:
        y_test, p_test = predict(model, test_loader, device)
        # **用验证集搜出来的阈值**，绝不在测试集上重搜 —— 那是作弊
        result.test_scores = evaluate_tagging(y_test, p_test, result.thresholds, tag_groups)

    return result


# --------------------------------------------------------------------------------------
# 结果记录
# --------------------------------------------------------------------------------------

def save_result(result: TrainResult, out: Path, tag_names: list[str] | None = None,
                save_model: bool = False, features: dict | None = None) -> None:
    """写结果 JSON；``save_model=True`` 时另存权重到同名 ``.pt``。

    权重默认**不存** —— 5 种子 × 十几个配置会堆出上百个 checkpoint，
    而其中绝大多数只是用来算方差的，存下来没有意义。
    只有要拿去做检索或部署的那一个才值得落盘。
    """
    out.parent.mkdir(parents=True, exist_ok=True)

    if save_model and result.state_dict is not None:
        import torch

        ckpt = out.with_suffix(".pt")
        # features 记录这个权重吃的是**哪份特征**（基座/层/段数/帧步长）。
        # 第一版没存：best_model.pt 的 config 里只有 pooling 和 loss，
        # 说不出自己是用第 6 层 4 段特征训的。配错特征加载**不会报错**，
        # 只会静默给出错误的预测 —— 形状对得上的特征多的是。
        torch.save({"state_dict": result.state_dict,
                    "config": result.config,
                    "features": features or {},
                    "thresholds": result.thresholds,
                    "tag_names": tag_names}, ckpt)
        print(f"  权重 → {ckpt}")

    def dump(s: TaggingScores | None) -> dict | None:
        if s is None:
            return None
        d = {k: v for k, v in vars(s).items() if k != "groups"}
        d["groups"] = {g: {k: v for k, v in vars(gs).items() if k != "groups"}
                       for g, gs in s.groups.items()}
        return d

    payload = {
        "level": result.level,
        "config": result.config,
        "features": features or {},
        "n_params": result.n_params,
        "best_epoch": result.best_epoch,
        "train_seconds": result.train_seconds,
        "history": [asdict(h) for h in result.history],
        "val": dump(result.val_scores),
        "test": dump(result.test_scores),
    }
    if result.thresholds is not None:
        payload["thresholds"] = {
            "median": float(np.median(result.thresholds)),
            "min": float(result.thresholds.min()),
            "max": float(result.thresholds.max()),
            # 阈值本身值得留档：如果搜出来的大量阈值贴着网格边界，
            # 说明模型的概率标定有问题，而不是阈值优化"很有效"
            "per_tag": (dict(zip(tag_names, result.thresholds.round(3).tolist(), strict=True))
                        if tag_names else result.thresholds.round(3).tolist()),
        }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
