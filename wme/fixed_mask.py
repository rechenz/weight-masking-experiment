"""
固定权重 Mask 约束
=================
与 ActivationConstraint（每次随机）和 SubspaceConstraint（SVD 截断）不同，
FixedWeightMask 在约束期开始时选择一个固定的权重子集进行 mask，
整个约束期内 mask 集合只减不增（逐步解封）。

对比意义：
- vs 随机 mask（每 batch 重选）：固定 mask 让被保留的权重稳定积累梯度
- vs dropout（激活级随机）：权重级固定短路 vs 激活级随机短路
- vs subspace SVD（结构最优截断）：无结构硬截断 vs 基于本征结构的最优截断

实现方式：直接 in-place mul_，无需 backup/restore。
被 mask 的权重 = 0 → 梯度 = 0 → optimizer 不碰 → 解锁后从零开始发育。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class FixedWeightMask:
    """
    固定权重 mask：约束期开始时生成 mask，全程不变。

    使用 in-place weight.mul_(mask)，梯度自然通过 mask 传播。
    无 backup/restore，避免梯度-权重错位导致的训练发散。
    """

    def __init__(
        self,
        model: nn.Module,
        max_rate: float = 0.5,
        total_epochs: int = 200,
        constraint_epochs: int = 100,
        schedule: str = "decay",
        seed: int = 42,
    ) -> None:
        self.model = model
        self.max_rate = max_rate
        self.total_epochs = total_epochs
        self.constraint_epochs = constraint_epochs
        self.schedule = schedule
        self.current_epoch: int = 0

        # 收集所有 Linear 层
        self._linear_layers: list[tuple[str, nn.Linear]] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                self._linear_layers.append((name, module))

        # 为每个 Linear 层生成固定的二进制 mask 并移到 GPU
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._masks: dict[str, torch.Tensor] = {}  # 全量 mask（50% zero）
        self._masked_indices: dict[str, torch.Tensor] = {}  # mask 位置的有序列表
        self._prev_rate: float = self.max_rate  # 上一 epoch 的 mask rate

        for name, layer in self._linear_layers:
            W = layer.weight.data
            mask = torch.ones_like(W, device=W.device)
            n_total = mask.numel()
            n_mask = int(self.max_rate * n_total)
            if n_mask > 0:
                indices = torch.randperm(n_total, generator=rng)[:n_mask]
                mask.view(-1)[indices] = 0
                self._masked_indices[name] = indices
            self._masks[name] = mask

        print(f"  [FixedWeightMask] {len(self._linear_layers)} 个 Linear 层 | "
              f"mask_rate={self.max_rate:.0%} (固定)")

    @property
    def mode(self) -> str:
        return "fixed_mask"

    def get_current_rate(self) -> float:
        """返回当前 mask 比例 (0~1)。"""
        if self.current_epoch >= self.constraint_epochs:
            return 0.0
        return self._get_mask_rate()

    def _get_mask_rate(self) -> float:
        """根据 schedule 计算当前 mask 比例。"""
        if self.current_epoch >= self.constraint_epochs:
            return 0.0

        progress = self.current_epoch / max(self.constraint_epochs, 1)

        if self.schedule in ("linear", "decay"):
            return self.max_rate * (1.0 - progress)
        elif self.schedule == "cosine":
            return self.max_rate * 0.5 * (1.0 + np.cos(np.pi * progress))
        elif self.schedule == "step":
            return self.max_rate
        elif self.schedule == "inverse":
            return self.max_rate * max(0.0, 1.0 - progress ** 2)
        elif self.schedule == "triangle":
            start = min(self.max_rate * 0.1, 0.05)
            mid = 0.5
            if progress <= mid:
                p = progress / mid
                return start + (self.max_rate - start) * p
            else:
                p = (progress - mid) / (1.0 - mid)
                return self.max_rate * (1.0 - p)
        return self.max_rate

    def apply_constraint(self) -> None:
        """
        直接对权重施加 in-place mask: W = W * mask。
        无 backup/restore —— optimizer 直接更新被 mask 的权重（梯度=0所以不动）。

        mask 随 schedule 衰减：只解封新位置，已解封的不再 mask。
        """
        mask_rate = self._get_mask_rate()
        if mask_rate <= 0 and self._prev_rate <= 0:
            return  # 无约束且之前也无约束，跳过

        for name, layer in self._linear_layers:
            full_mask = self._masks[name]  # 已在 GPU 上
            if mask_rate <= 0:
                # 约束结束，不再 mask
                continue
            if mask_rate >= self.max_rate - 1e-8:
                # 全量 mask
                layer.weight.data.mul_(full_mask)
            else:
                # 渐进解封：从全量 mask 中取前 n_mask 个保持 mask
                indices = self._masked_indices[name]
                n_mask = int(mask_rate * full_mask.numel())
                keep_masked = indices[:n_mask]
                active_mask = torch.ones_like(full_mask)
                active_mask.view(-1)[keep_masked] = 0
                layer.weight.data.mul_(active_mask)

        self._prev_rate = mask_rate

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch。"""
        self.current_epoch = epoch
