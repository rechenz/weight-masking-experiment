"""
子空间约束模块 — SVD 低秩截断
=============================
类比人类婴儿"机能不全→健全"的数学实现：
训练前期将权重限制在低秩子空间（短路未发育的维度），
后期逐步放开到全秩。

核心思路：
- 每 epoch 对所有权重矩阵做 SVD 分解
- 训练时用截断的低秩重建直接替代原始权重（in-place）
- 无 backup/restore —— optimizer 直接更新截断后的权重
- 下个 epoch 重新 SVD，反映权重更新
- 约束阶段结束后，权重回到全秩，模型正常训练
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class SubspaceConstraint:
    """
    SVD 截断约束：限定权重矩阵在低秩子空间中运行。

    与 ActivationConstraint（dropout/噪声）不同，
    这是真正的"数学短路"——被截断的维度完全不存在于计算图中。

    约束阶段结束后，所有权重恢复全秩，模型正常训练。
    """

    def __init__(
        self,
        model: nn.Module,
        max_rank_ratio: float = 0.3,
        total_epochs: int = 200,
        constraint_epochs: int = 100,
        schedule: str = "decay",
    ) -> None:
        self.model = model
        self.max_rank_ratio = max_rank_ratio
        self.total_epochs = total_epochs
        self.constraint_epochs = constraint_epochs
        self.schedule = schedule
        self.current_epoch: int = 0

        # 收集所有 Linear 层
        self._linear_layers: list[tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_layers.append((name, module))

        # 缓存 SVD 分解（每 epoch 更新）
        self._svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        print(f"  [Subspace] 检测到 {len(self._linear_layers)} 个 Linear 层")
        self._update_svd()

    @property
    def mode(self) -> str:
        return "subspace"

    def get_current_rate(self) -> float:
        """返回当前 rank 比率 (0~1)。"""
        return self._get_rank_ratio()

    def _get_rank_ratio(self) -> float:
        """根据 schedule 计算当前 epoch 的 rank 保留比例。"""
        if self.current_epoch >= self.constraint_epochs:
            return 1.0

        progress = self.current_epoch / max(self.constraint_epochs, 1)
        min_ratio = 0.05  # 最低保留 5% rank

        if self.schedule in ("linear", "decay"):
            return min_ratio + (1.0 - min_ratio) * progress
        elif self.schedule == "cosine":
            return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 - np.cos(np.pi * progress))
        elif self.schedule == "step":
            return self.max_rank_ratio if progress < 0.5 else 1.0
        elif self.schedule == "inverse":
            return min_ratio + (1.0 - min_ratio) * (1.0 - (1.0 - progress) ** 3)
        elif self.schedule == "triangle":
            mid = 0.5
            if progress <= mid:
                p = progress / mid
                return min_ratio + (self.max_rank_ratio - min_ratio) * p
            else:
                p = (progress - mid) / (1.0 - mid)
                return self.max_rank_ratio + (1.0 - self.max_rank_ratio) * p
        return self.max_rank_ratio

    def _update_svd(self) -> None:
        """对所有权重矩阵重新计算 SVD。每 epoch 开始时调用。"""
        for name, layer in self._linear_layers:
            W = layer.weight.data.float()
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                self._svd_cache[name] = (U, S, Vh)
            except Exception:
                # SVD 失败时保留旧缓存，跳过更新
                pass

    def apply_constraint(self) -> None:
        """
        将权重替换为截断 SVD 重建（in-place，无 restore）。
        optimizer 直接更新截断后的权重。
        """
        rank_ratio = self._get_rank_ratio()
        if rank_ratio >= 1.0:
            return  # 全秩，无需截断

        for name, layer in self._linear_layers:
            if name not in self._svd_cache:
                continue
            U, S, Vh = self._svd_cache[name]
            k = max(1, int(len(S) * rank_ratio))
            # 截断重建: W_k = U[:,:k] @ diag(S[:k]) @ Vh[:k,:]
            W_trunc = (U[:, :k] * S[:k].unsqueeze(0)) @ Vh[:k, :]
            # 直接替换，无 backup/restore
            layer.weight.data.copy_(W_trunc.to(layer.weight.dtype))

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，触发 SVD 更新。"""
        self.current_epoch = epoch
        if epoch <= self.constraint_epochs:
            self._update_svd()
