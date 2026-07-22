"""
固定权重 Mask 约束
=================
与 ActivationConstraint（每次随机）和 SubspaceConstraint（SVD 截断）不同，
FixedWeightMask 在约束期开始时选择一个固定的权重子集进行 mask，
整个约束期内 mask 不发生变化。

对比意义：
- vs 随机 mask（每 batch 重选）：固定 mask 让被保留的权重稳定积累梯度
- vs dropout（激活级随机）：权重级固定短路 vs 激活级随机短路
- vs subspace SVD（结构最优截断）：无结构硬截断 vs 基于本征结构的最优截断

假设：
- 被 mask 的权重虽然前向被清零，但反向能积累梯度
- 解锁时它们已经"暗中发育"，不至于从零开始
- 固定的"机能不全"比随机稳定性更好
"""

import torch
import torch.nn as nn
from typing import Optional


class FixedWeightMask:
    """
    固定权重 mask：约束期开始时生成 mask，全程不变。

    约束阶段结束后，mask 移除，所有权重正常训练。
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

        # 为每个 Linear 层生成固定的二进制 mask
        rng = torch.Generator()
        rng.manual_seed(seed)
        self._masks: dict[str, torch.Tensor] = {}
        for name, layer in self._linear_layers:
            W = layer.weight.data
            mask = torch.ones_like(W, device="cpu")
            n_total = mask.numel()
            n_mask = int(self.max_rate * n_total)
            if n_mask > 0:
                indices = torch.randperm(n_total, generator=rng)[:n_mask]
                mask.view(-1)[indices] = 0
            self._masks[name] = mask

        # 权重备份
        self._backup: dict[str, torch.Tensor] = {}

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

        import numpy as np
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
        对权重施加固定 mask: W = W * mask。
        mask 比例随 schedule 变化：
        - step: 始终用原始 mask（固定 mask_rate）
        - decay/linear: 按 schedule 降低 mask 比例，从全量 mask 中取子集
        """
        mask_rate = self._get_mask_rate()
        if mask_rate <= 0:
            return

        for name, layer in self._linear_layers:
            full_mask = self._masks[name].to(layer.weight.device)
            if mask_rate < self.max_rate - 1e-8:
                # schedule 衰减: 只 mask full_mask 中前 mask_rate/max_rate 比例的 0
                # 即渐进式"解封"被 mask 的权重
                n_mask = int(mask_rate * full_mask.numel())
                # 用 full_mask 中值为 0 的位置（即原始被 mask 的），只取前 n_mask 个保持 mask
                zero_positions = (full_mask.view(-1) == 0).nonzero(as_tuple=True)[0]
                keep_masked = zero_positions[:n_mask]
                active_mask = torch.ones_like(full_mask.view(-1))
                active_mask[keep_masked] = 0
                active_mask = active_mask.view_as(full_mask)
            else:
                active_mask = full_mask

            # 备份 + 施加
            self._backup[name] = layer.weight.data.clone()
            layer.weight.data.copy_(layer.weight.data * active_mask)

    def restore_weights(self) -> None:
        """恢复原始权重。在 backward 之后调用。"""
        for name, layer in self._linear_layers:
            if name in self._backup:
                layer.weight.data.copy_(self._backup[name])
        self._backup.clear()

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch。"""
        self.current_epoch = epoch
