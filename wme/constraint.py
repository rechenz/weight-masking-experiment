"""
激活层约束策略模块
=================
用 duck-typing 的方式对任意实现了 set_dropout_rate / set_actnoise_std 接口的模型施加约束。
"""

import numpy as np
from typing import Optional
import torch.nn as nn


class ActivationConstraint:
    """
    训练期间对模型施加激活层约束（非权重层面）。

    通过调整模型的 dropout 率和激活噪声幅度，
    模拟「感官/处理不成熟→逐渐健全」的过程。
    兼容 ResNet、SmallCNN、TinyViT 等所有实现了
    set_dropout_rate / set_actnoise_std 接口的模型。
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "dropout",
        max_rate: float = 0.5,
        total_epochs: int = 200,
        constraint_epochs: int = 100,
        schedule: str = "decay",
    ) -> None:
        self.model = model
        self.mode = mode
        self.max_rate = max_rate
        self.total_epochs = total_epochs
        self.constraint_epochs = constraint_epochs
        self.schedule = schedule
        self.current_epoch: int = 0

    def get_current_rate(self) -> float:
        """根据 schedule 返回当前 epoch 的实际约束强度。
        返回 0~1 之间的值，对应 dropout 率或噪声幅度缩放。
        """
        if self.current_epoch >= self.constraint_epochs:
            return 0.0  # 后期全部放开

        progress = self.current_epoch / max(self.constraint_epochs, 1)

        if self.schedule == "step":
            return self.max_rate
        elif self.schedule in ("linear", "decay"):
            return self.max_rate * (1 - progress)
        elif self.schedule == "cosine":
            return self.max_rate * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule == "inverse":
            return self.max_rate * max(0, 1 - progress ** 2)
        elif self.schedule == "triangle":
            start = min(self.max_rate * 0.1, 0.05)
            mid = 0.5
            if progress <= mid:
                p = progress / mid
                return start + (self.max_rate - start) * p
            else:
                p = (progress - mid) / (1.0 - mid)
                return self.max_rate * (1 - p)
        return self.max_rate

    def apply_constraint(self) -> None:
        """在每个 batch 训练前调用，更新模型的约束参数。"""
        rate = self.get_current_rate()
        if rate <= 0:
            self.model.set_dropout_rate(0.0)  # type: ignore[union-attr]
            self.model.set_actnoise_std(0.0)  # type: ignore[union-attr]
            return

        if self.mode == "dropout":
            self.model.set_dropout_rate(rate)  # type: ignore[union-attr]
            self.model.set_actnoise_std(0.0)  # type: ignore[union-attr]
        elif self.mode == "actnoise":
            self.model.set_dropout_rate(0.0)  # type: ignore[union-attr]
            # 噪声 std 从 0 到 max_rate * 0.3
            self.model.set_actnoise_std(rate * 0.3)  # type: ignore[union-attr]

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = epoch
