"""相似曲目检索：嵌入 + 精确检索 + IVF 近似检索。

**为什么不用 FAISS**：本库 5,404 首 × 768 维 = 15.8 MB，
精确检索就是一次矩阵乘（实测 < 5 ms）。为这个规模引一个重依赖是做样子。
IVF 自己实现（k-means 分桶 + 只搜最近的几个桶）只要几十行，
而且能直接产出**速度/召回权衡表** —— 那才是这一节真正要说明的东西：
*在什么规模上近似检索才开始划算*。

三种嵌入的对照是本模块的实验重点：

=================  ====  ================================================
嵌入               维度  含义
=================  ====  ================================================
``mert``           768   冻结基座特征的时间平均。**无监督**
``hidden``         512   标签头 MLP 的中间层。受 50 个标签**弱监督**
``logits``          50   标签头输出。监督信号最强，但维度极低
=================  ====  ================================================

它们回答同一个问题：**为标签任务训练的表征，检索效果会更好吗？**
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from src.eval.retrieval import l2_normalize


@dataclass
class SearchStats:
    """一次检索的耗时与访问量，用来做速度/召回权衡。"""

    seconds: float
    n_compared: int          # 实际算了多少个内积 —— 近似方法省的就是这个


class ExactIndex:
    """精确最近邻（余弦相似度）。

    嵌入先 L2 归一化，于是内积 == 余弦相似度，检索退化成一次矩阵乘。
    """

    def __init__(self, embeddings: np.ndarray):
        self.emb = l2_normalize(np.asarray(embeddings, dtype=np.float32))

    def search(self, queries: np.ndarray, k: int,
               exclude_self: bool = True) -> tuple[np.ndarray, SearchStats]:
        q = l2_normalize(np.asarray(queries, dtype=np.float32))
        t0 = time.perf_counter()
        sim = q @ self.emb.T
        if exclude_self:
            # 查询就在库里时，第一名永远是它自己 —— 必须排除，
            # 否则 Recall@K 会凭空多一个"命中"
            n = min(len(q), len(self.emb))
            sim[np.arange(n), np.arange(n)] = -np.inf
        idx = np.argpartition(-sim, k, axis=1)[:, :k]
        rows = np.arange(len(q))[:, None]
        idx = idx[rows, np.argsort(-sim[rows, idx], axis=1)]
        return idx, SearchStats(time.perf_counter() - t0, len(q) * len(self.emb))


class IVFIndex:
    """倒排文件索引：k-means 分桶，查询只搜最近的 ``n_probe`` 个桶。

    这是 FAISS ``IndexIVFFlat`` 的最小实现。**召回不再是 100%** ——
    真正的近邻可能落在没被搜到的桶里，所以必须报
    :func:`src.eval.retrieval.index_recall`（近似结果与精确结果的重合率），
    否则"快了 10 倍"这句话没有意义。
    """

    def __init__(self, embeddings: np.ndarray, n_lists: int = 64, seed: int = 0):
        from sklearn.cluster import KMeans

        self.emb = l2_normalize(np.asarray(embeddings, dtype=np.float32))
        n_lists = max(1, min(n_lists, len(self.emb) // 2))
        km = KMeans(n_clusters=n_lists, random_state=seed, n_init=4).fit(self.emb)
        self.centroids = l2_normalize(km.cluster_centers_.astype(np.float32))
        self.assign = km.labels_
        self.buckets = [np.flatnonzero(self.assign == c) for c in range(n_lists)]

    def search(self, queries: np.ndarray, k: int, n_probe: int = 4,
               exclude_self: bool = True) -> tuple[np.ndarray, SearchStats]:
        q = l2_normalize(np.asarray(queries, dtype=np.float32))
        t0 = time.perf_counter()
        probe = np.argsort(-(q @ self.centroids.T), axis=1)[:, :n_probe]

        out = np.full((len(q), k), -1, dtype=int)
        compared = 0
        for i in range(len(q)):
            cand = np.concatenate([self.buckets[c] for c in probe[i]]) if n_probe else np.array([])
            if exclude_self:
                cand = cand[cand != i]
            if len(cand) == 0:
                continue
            sim = self.emb[cand] @ q[i]
            compared += len(cand)
            top = np.argsort(-sim)[:k]
            out[i, : len(top)] = cand[top]
        return out, SearchStats(time.perf_counter() - t0, compared)


def embed_from_features(paths: list, pool: str = "mean") -> np.ndarray:
    """把逐帧特征缓存池化成每首一个向量。"""
    out = []
    for p in paths:
        x = np.load(p).astype(np.float32)
        out.append(x.mean(axis=0) if pool == "mean" else x.max(axis=0))
    return np.stack(out)


def embed_with_head(paths: list, ckpt_path, layer: str = "hidden",
                    device: str = "auto") -> np.ndarray:
    """用训练好的标签头产生嵌入。

    Args:
        layer: ``"hidden"`` 取 MLP 中间层（512 维），``"logits"`` 取输出（50 维）
    """
    import torch

    from src.tagging.backbone import pick_device
    from src.tagging.models import AttentionPoolHead

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    dev = pick_device(device)
    model = AttentionPoolHead(dim=768, n_tags=len(ck["tag_names"]),
                              pooling=cfg.get("pooling") or "mean")
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()

    # hidden 要的是 MLP 第 3 层（Linear 768→512）之后、GELU 之后的激活。
    # 用 forward hook 取，不改模型结构 —— 改结构会让加载权重时对不上。
    feats: list = []
    if layer == "hidden":
        target = model.mlp[3]                      # GELU
        h = target.register_forward_hook(lambda m, i, o: feats.append(o.detach().cpu()))

    out = []
    with torch.no_grad():
        for p in paths:
            x = torch.from_numpy(np.load(p).astype(np.float32))[None].to(dev)
            logits = model(x)
            if layer == "logits":
                out.append(logits[0].cpu().numpy())
            else:
                out.append(feats.pop()[0].numpy())
    if layer == "hidden":
        h.remove()
    return np.stack(out)
