"""
Weight Masking Experiment
=======================
类比人类婴儿"机能不全→健全"的训练策略验证。

模块拆分:
- models: 模型定义 (ResNet, SmallCNN, TinyViT)
- constraint: 激活层约束策略
- training: 训练与评估循环
- utils: 工具函数
- experiment: 主实验与绘图
"""

from wme.models import (
    BasicBlock, ResNet, resnet18,
    SmallCNN,
    PatchEmbed, TransformerBlock, TinyViT,
    create_model,
    ConstraintModel,
)
from wme.constraint import ActivationConstraint
from wme.training import train_epoch, evaluate, count_model_params
from wme.utils import _convert_to_json_safe
from wme.experiment import run_experiment, draw_comparison

__all__ = [
    "BasicBlock", "ResNet", "resnet18",
    "SmallCNN",
    "PatchEmbed", "TransformerBlock", "TinyViT",
    "create_model", "ConstraintModel",
    "ActivationConstraint",
    "train_epoch", "evaluate", "count_model_params",
    "_convert_to_json_safe",
    "run_experiment", "draw_comparison",
]
