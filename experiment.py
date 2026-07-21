#!/usr/bin/env python -u
"""
Weight Masking Experiment
=======================
类比人类婴儿"机能不全→健全"的训练策略验证。

策略：训练前期随机屏蔽部分权重（模拟机能不全），后期放开（模拟机能健全）。
对比正常训练，看对收敛、泛化、创造性的影响。

实验设计：
- 数据集: CIFAR-10
- 模型: ResNet-18 (CIFAR-10 适配版)
- 策略: 多种 mask 方案对比
- 指标: loss 曲线、测试准确率、泛化 gap

作者: 银狼 (SilverWolf) — 热尘不想写代码的时候我写的
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from copy import deepcopy
from typing import Optional, Callable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================================
# 模型定义 — ResNet-18 for CIFAR-10
# ============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=3):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet18(num_classes=10, in_channels=3):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


# ============================================================================
# 轻量模型 — SmallCNN（省电模式专用）
# ============================================================================

class SmallCNN(nn.Module):
    """轻量 CNN，~250K 参数，GPU 负载只有 ResNet-18 的 1/40。"""
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# 小 Transformer — TinyViT（跟上时代）
# ============================================================================

class PatchEmbed(nn.Module):
    """图像分块 + 线性投影。"""
    def __init__(self, img_size=32, patch_size=4, in_chans=1, embed_dim=192):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.forward_dropout_rate = 0.0  # 运行时动态调整

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + F.dropout(self.mlp(self.norm2(x)),
                          p=self.forward_dropout_rate, training=self.training)
        return x


class TinyViT(nn.Module):
    """超轻量 ViT，适合 32x32 小图，~2.5M 参数。"""
    def __init__(self, img_size=32, patch_size=4, in_channels=1, num_classes=10,
                 embed_dim=192, depth=6, num_heads=6, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.actnoise_std = 0.0  # 运行时动态调整

        self._init_weights()

    def set_dropout_rate(self, rate: float):
        """动态调整所有 block 的 dropout 率。"""
        for block in self.blocks:
            block.forward_dropout_rate = rate

    def set_actnoise_std(self, std: float):
        """动态调整激活噪声幅度。"""
        self.actnoise_std = std

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(lambda m: (nn.init.trunc_normal_(m.weight, std=0.02)
                               if isinstance(m, (nn.Linear, nn.Conv2d)) else
                               nn.init.constant_(m.bias, 0) if hasattr(m, "bias") and m.bias is not None else None))

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        # 激活噪声（在 head 之前加，影响最后的线性层输入）
        if self.actnoise_std > 0 and self.training:
            x = x + torch.randn_like(x) * self.actnoise_std
        x = self.head(x)
        return x


def create_model(name: str, num_classes: int, in_channels: int) -> nn.Module:
    if name == "resnet18":
        return resnet18(num_classes, in_channels)
    elif name == "smallcnn":
        return SmallCNN(in_channels, num_classes)
    elif name == "vit":
        return TinyViT(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


# ============================================================================
# 权重 Mask 策略集合
# ============================================================================

def get_weight_params(model: nn.Module, exclude_bias_bn: bool = True) -> List[torch.Tensor]:
    """获取模型中所有可 mask 的权重参数（排除 bias 和 BN）。"""
    params = []
    for name, p in model.named_parameters():
        if exclude_bias_bn and ("bias" in name or "bn" in name or "shortcut.1" in name):
            continue
        if p.dim() >= 2:  # 只 mask 权重矩阵/卷积核，不 mask 标量
            params.append(p)
    return params


# ============================================================================
# 激活层约束策略（替代权重 mask）
# ============================================================================

class ConstraintMode:
    """存储约束模式标识符。"""
    NONE = "none"            # Baseline: 正常训练
    DROPOUT = "dropout"      # 阶段性高 dropout → 降低到 0
    ACTNOISE = "actnoise"    # 激活值加高斯噪声 → 降低到 0


class ActivationConstraint:
    """
    训练期间对模型施加激活层约束（非权重层面）。

    通过调整模型的 dropout 率和激活噪声幅度，
    模拟「感官/处理不成熟→逐渐健全」的过程。
    """

    def __init__(
        self,
        model: TinyViT,
        mode: str = "dropout",
        max_rate: float = 0.5,
        total_epochs: int = 200,
        constraint_epochs: int = 100,
        schedule: str = "decay",
    ):
        self.model = model
        self.mode = mode
        self.max_rate = max_rate
        self.total_epochs = total_epochs
        self.constraint_epochs = constraint_epochs
        self.schedule = schedule
        self.current_epoch = 0

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
            start = 0.05
            mid = 0.5
            if progress <= mid:
                p = progress / mid
                return start + (self.max_rate - start) * p
            else:
                p = (progress - mid) / (1.0 - mid)
                return self.max_rate * (1 - p)
        return self.max_rate

    def apply_constraint(self):
        """在每个 batch 训练前调用，更新模型的约束参数。"""
        rate = self.get_current_rate()
        if rate <= 0:
            self.model.set_dropout_rate(0.0)
            self.model.set_actnoise_std(0.0)
            return

        if self.mode == "dropout":
            self.model.set_dropout_rate(rate)
            self.model.set_actnoise_std(0.0)
        elif self.mode == "actnoise":
            self.model.set_dropout_rate(0.0)
            # 噪声 std 从 0 到 max_rate * 0.3
            self.model.set_actnoise_std(rate * 0.3)

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch


# ============================================================================
# 训练与评估
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    constraint: Optional[ActivationConstraint] = None,
    log_interval: int = 50,
    epoch: int = 0,
    use_amp: bool = False,
    scaler=None,
) -> Tuple[float, float]:
    """训练一个 epoch，返回 (avg_loss, accuracy)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()

        # 训练前施加激活层约束
        if constraint is not None:
            constraint.apply_constraint()

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0:
            cr = constraint.get_current_rate() if constraint else 0
            print(f"  [{epoch}][{batch_idx:3d}/{len(loader):3d}] "
                  f"loss: {loss.item():.4f} "
                  f"acc: {100. * correct / total:.2f}%"
                  + (f" constraint: {cr:.3f}" if constraint else ""))

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """评估模型，返回 (avg_loss, accuracy)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def count_model_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# 主实验函数
# ============================================================================

def run_experiment(
    config: dict,
    device: torch.device,
    train_loader: data.DataLoader,
    test_loader: data.DataLoader,
    save_dir: Path,
) -> dict:
    """
    运行一次实验，返回结果字典。
    config 包含实验配置，包括 mask 策略参数。
    """
    experiment_name = config["name"]
    print(f"\n{'='*60}")
    print(f"  实验: {experiment_name}")
    print(f"{'='*60}")

    # 初始化模型
    model = create_model(config.get("model", "resnet18"),
                      num_classes=10,
                      in_channels=config.get("in_channels", 3)).to(device)
    print(f"  模型参数量: {count_model_params(model):,}")

    # 损失函数和优化器 — 根据模型类型自动选择
    criterion = nn.CrossEntropyLoss()
    model_type = config.get("model", "resnet18")
    if "vit" in model_type:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.05),
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.get("lr", 0.1),
            momentum=config.get("momentum", 0.9),
            weight_decay=config.get("weight_decay", 5e-4),
            nesterov=True,
        )

    # 学习率调度器 — ViT 加 warmup
    if "vit" in model_type:
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10),
                optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"] - 10, eta_min=1e-5),
            ],
            milestones=[10],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=1e-4
        )

    # 初始化激活层约束（替代权重 mask）
    constraint = None
    if config["constraint_mode"] != "none":
        constraint = ActivationConstraint(
            model=model,
            mode=config["constraint_mode"],
            max_rate=config.get("max_rate", 0.5),
            total_epochs=config["epochs"],
            constraint_epochs=config.get("constraint_epochs", config["epochs"] // 2),
            schedule=config.get("constraint_schedule", "decay"),
        )

    use_amp = config.get("amp", False)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # 训练日志
    log = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "constraint_rate": [],
        "lr": [],
        "config": config,
    }

    best_acc = 0.0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        if constraint is not None:
            constraint.set_epoch(epoch)

        print(f"\n--- Epoch {epoch}/{config['epochs']} "
              f"| lr: {scheduler.get_last_lr()[0]:.6f} "
              + (f"| constraint: {constraint.get_current_rate() if constraint else 0:.3f}"))

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            constraint=constraint, epoch=epoch, use_amp=use_amp, scaler=scaler,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["constraint_rate"].append(constraint.get_current_rate() if constraint else 0)
        log["lr"].append(scheduler.get_last_lr()[0])

        print(f"  >>> Train loss: {train_loss:.4f}  acc: {train_acc:.2f}%")
        print(f"  >>> Test  loss: {test_loss:.4f}  acc: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = deepcopy(model.state_dict())
            torch.save(best_state, save_dir / f"best_{experiment_name}.pt")
            print(f"  ✨ 新最佳: {best_acc:.2f}%")

    # 最终评估
    final_loss, final_acc = test_loss, test_acc
    print(f"\n  🔚 最终结果: test_acc={final_acc:.2f}%  |  best_acc={best_acc:.2f}%")

    log["final_test_acc"] = final_acc
    log["best_test_acc"] = best_acc

    return log


def draw_comparison(results: dict, save_dir: Path):
    """绘制对比图。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = {
        "none": "#2196F3",
        "dropout": "#F44336",
        "actnoise": "#4CAF50",
    }
    labels = {
        "none": "Baseline",
        "dropout": "Scheduled Dropout",
        "actnoise": "Activation Noise",
    }
    linestyles = {
        "none": "-",
        "dropout": "--",
        "actnoise": "-.",
    }

    for (name, log) in results.items():
        cmode = log["config"].get("constraint_mode", log["config"]["mask_mode"])
        color = colors.get(cmode, "#666")
        ls = linestyles.get(cmode, "-")
        label = labels.get(cmode, name)
        sched = log["config"].get("constraint_schedule", "")
        if sched:
            label += f" ({sched})"
        epochs = range(1, len(log["test_acc"]) + 1)

        constraint_epochs = log["config"].get("constraint_epochs", log["config"]["epochs"] // 2)

        # 在图上画约束切换线
        for ax in axes.flat:
            ax.axvline(x=constraint_epochs, color="gray", linestyle="--",
                       alpha=0.3, label="_constraint_boundary")

        # Test accuracy
        axes[0, 0].plot(epochs, log["test_acc"], color=color, ls=ls,
                        lw=2, label=label)
        # Train accuracy
        axes[0, 1].plot(epochs, log["train_acc"], color=color, ls=ls,
                        lw=2, label=label)
        # Test loss
        axes[1, 0].plot(epochs, log["test_loss"], color=color, ls=ls,
                        lw=2, label=label)
        # Train loss
        axes[1, 1].plot(epochs, log["train_loss"], color=color, ls=ls,
                        lw=2, label=label)

    axes[0, 0].set_title("Test Accuracy (%)", fontsize=13)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=9, loc="lower right")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title("Train Accuracy (%)", fontsize=13)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(fontsize=9, loc="lower right")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title("Test Loss", fontsize=13)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend(fontsize=9, loc="upper right")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title("Train Loss", fontsize=13)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].legend(fontsize=9, loc="upper right")
    axes[1, 1].grid(True, alpha=0.3)

    # constraint_rate 和 lr 图
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    for (name, log) in results.items():
        cmode = log["config"].get("constraint_mode", log["config"]["mask_mode"])
        color = colors.get(cmode, "#666")
        sched = log["config"].get("constraint_schedule", "")
        label = labels.get(cmode, name)
        if sched:
            label += f" ({sched})"
        epochs = range(1, len(log["constraint_rate"]) + 1)
        axes2[0].plot(epochs, log["constraint_rate"], color=color, lw=2, label=label)
        axes2[1].plot(epochs, log["lr"], color=color, lw=2, label=label)

    axes2[0].set_title("Constraint Rate (per epoch)", fontsize=13)
    axes2[0].set_xlabel("Epoch")
    axes2[0].legend(fontsize=9)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_title("Learning Rate", fontsize=13)
    axes2[1].set_xlabel("Epoch")
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "comparison_accuracy_loss.png", dpi=150, bbox_inches="tight")
    fig2.savefig(save_dir / "constraint_lr_schedule.png", dpi=150, bbox_inches="tight")
    print(f"  图片已保存: {save_dir}/comparison_accuracy_loss.png")
    print(f"  图片已保存: {save_dir}/constraint_lr_schedule.png")
    plt.close("all")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Activation Constraint Experiment")
    parser.add_argument("--epochs", type=int, default=100,
                        help="总训练 epoch 数 (默认: 100)")
    parser.add_argument("--constraint-epochs", type=int, default=None,
                        help="前多少 epoch 施加约束 (默认: epochs//2)")
    parser.add_argument("--max-rate", type=float, default=0.5,
                        help="最大约束强度 — dropout率/噪声幅度 (默认: 0.5)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="batch size (默认: 64)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="初始学习率 (默认: 0.001)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader worker 数 (默认: 2)")
    parser.add_argument("--modes", type=str, nargs="+",
                        choices=["none", "dropout", "actnoise"],
                        default=["none", "dropout", "actnoise"],
                        help="约束模式 (默认: 全部)")
    parser.add_argument("--schedules", type=str, nargs="+", default=["decay"],
                        choices=["step", "linear", "decay", "cosine", "inverse", "triangle"],
                        help="约束 schedule 列表 (默认: decay)")
    parser.add_argument("--dataset", type=str, default="fashion_mnist",
                        choices=["cifar10", "fashion_mnist"],
                        help="数据集 (默认: fashion_mnist)")
    parser.add_argument("--model", type=str, default="vit",
                        choices=["resnet18", "smallcnn", "vit"],
                        help="模型 (默认: vit)")
    parser.add_argument("--amp", action="store_true",
                        help="启用混合精度训练 (FP16)")
    parser.add_argument("--light", action="store_true",
                        help="轻量模式: epochs=50 + batch=64 + amp")
    args = parser.parse_args()

    # --light 覆盖默认配置
    if args.light:
        if args.epochs == 100:
            args.epochs = 50
        args.amp = True

    # 固定随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device} | CUDA: {torch.cuda.is_available()}")
    print(f"Torch: {torch.__version__} | Python: {sys.version}")

    # 数据加载
    if args.dataset == "cifar10":
        print("\n加载 CIFAR-10...")
        in_channels = 3
        num_classes = 10
        img_size = 32
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test)
    else:
        print("\n加载 FashionMNIST...")
        in_channels = 1
        num_classes = 10
        img_size = 28
        # F-MNIST 28x28 -> 需要 resize 到 32x32（适应 ResNet 结构）
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ])
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform_train)
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform_test)

    print(f"  数据集: {args.dataset} | 输入通道: {in_channels} | 类别: {num_classes}")
    print(f"  训练集: {len(train_dataset)}  | 测试集: {len(test_dataset)}")

    train_loader = data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    test_loader = data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 显示实验配置
    epochs = args.epochs
    constraint_epochs = args.constraint_epochs if args.constraint_epochs else epochs // 2

    base_config = {
        "epochs": epochs,
        "constraint_epochs": constraint_epochs,
        "max_rate": args.max_rate,
        "lr": args.lr,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataset": args.dataset,
        "model": args.model,
        "in_channels": in_channels,
        "amp": args.amp,

    print(f"\n实验配置:")
    print(f"  Epochs: {epochs} | Constraint Epochs: {constraint_epochs} | Max Rate: {args.max_rate}")
    print(f"  Schedules: {args.schedules} | Model: {args.model}")
    print(f"  Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  模式: {args.modes}")

    # 计算结果保存路径
    save_dir = Path(f"results_{args.seed}")
    save_dir.mkdir(exist_ok=True)
    print(f"\n结果保存: {save_dir.resolve()}")

    # 运行实验
    results = {}

    # 先跑一个 baseline（schedule 无关）
    cfg = {**base_config, "name": "constraint_none", "constraint_mode": "none"}
    log = run_experiment(cfg, device, train_loader, test_loader, save_dir)
    results["mask_none"] = log
    with open(save_dir / "log_none.json", "w") as f:
        clean = {k: (v if not isinstance(v, list) else
                    [float(x) if not isinstance(x, (int, float)) else x for x in v])
                for k, v in log.items() if k != "config"}
        clean["config"] = {k: v for k, v in log["config"].items() if isinstance(v, (str, int, float, bool))}
        json.dump(clean, f, indent=2)

    # 再各个 schedule 跑 mask 实验
    for schedule in args.schedules:
        for mode in args.modes:
            if mode == "none":
                continue  # baseline 已经跑过了
            name = f"constraint_{mode}_{schedule}"
            cfg = {**base_config, "name": name, "constraint_mode": mode, "constraint_schedule": schedule}
            log = run_experiment(cfg, device, train_loader, test_loader, save_dir)
            results[name] = log
            with open(save_dir / f"log_{mode}_{schedule}.json", "w") as f:
                clean = {k: (v if not isinstance(v, list) else
                            [float(x) if not isinstance(x, (int, float)) else x for x in v])
                        for k, v in log.items() if k != "config"}
                clean["config"] = {k: v for k, v in log["config"].items() if isinstance(v, (str, int, float, bool))}
                json.dump(clean, f, indent=2)

    # 画对比图 — 每个 schedule 一张
    schedules_to_plot = ["none"] + args.schedules if args.schedules != ["none"] else ["none"]
    for schedule in schedules_to_plot:
        subset = {k: v for k, v in results.items()
                  if schedule == "none" and "none" in k
                  or schedule in k}
        if len(subset) > 1:
            draw_comparison(subset, save_dir)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"  📊 实验总结")
    print(f"{'='*60}")
    for name, log in results.items():
        mode = log["config"]["constraint_mode"]
        sched = log["config"].get("constraint_schedule", "none")
        label = f"{mode:8s} ({sched})"
        print(f"  {label:20s} | Best Test Acc: {log['best_test_acc']:.2f}% | "
              f"Final: {log['final_test_acc']:.2f}%")

    print(f"\n✨ 全部完成!")


labels_map = {
    "none": "Baseline",
    "hard": "Hard Mask",
    "noise": "Noise Injection",
    "struct": "Structured Mask",
}


if __name__ == "__main__":
    main()
