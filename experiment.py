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


class MaskMode:
    """存储 mask 模式标识符，方便产出文件名。"""
    NONE = "none"            # Baseline: 正常训练
    HARD_BERNOULLI = "hard"  # 硬 mask：随机把权重置 0
    NOISE_GAUSS = "noise"    # 噪声注入：给权重加高斯噪声
    STRUCTURED = "struct"    # 结构化 mask：按输出通道整组 mask


class WeightMaskHook:
    """
    训练 Hook：在 optimizer.step() 之后对权重施加 mask / 噪声。

    支持多种 mask 策略和 schedule。
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "hard",
        mask_ratio: float = 0.5,
        total_epochs: int = 200,
        mask_epochs: int = 100,
        schedule: str = "step",
        noise_std: float = 0.01,
    ):
        self.model = model
        self.mode = mode
        self.mask_ratio = mask_ratio
        self.total_epochs = total_epochs
        self.mask_epochs = mask_epochs
        self.schedule = schedule
        self.noise_std = noise_std
        self.current_epoch = 0
        self.mask_steps = 0  # 累计 step 数

        # 收集可 mask 的参数
        self.target_params = get_weight_params(model)

    def get_current_mask_ratio(self) -> float:
        """根据 schedule 返回当前 epoch 的实际 mask 比例。"""
        if self.current_epoch >= self.mask_epochs:
            return 0.0  # 后期全部放开

        progress = self.current_epoch / max(self.mask_epochs, 1)

        if self.schedule == "step":
            # 分段常量：指定前 mask_epochs 固定比例
            return self.mask_ratio
        elif self.schedule == "linear":
            # 线性衰减：从 mask_ratio 递减到 0
            return self.mask_ratio * (1 - progress)
        elif self.schedule == "cosine":
            # 余弦衰减
            return self.mask_ratio * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.schedule == "inverse":
            # 前期大比例，快速衰减
            scale = max(0, 1 - progress ** 2)
            return self.mask_ratio * scale
        else:
            return self.mask_ratio

    def step(self):
        """在 optimizer.step() 后调用，对权重施加 mask。"""
        self.mask_steps += 1
        ratio = self.get_current_mask_ratio()

        if ratio <= 0:
            return

        with torch.no_grad():
            for p in self.target_params:
                if p.grad is None:
                    continue

                if self.mode == "hard":
                    # 硬 mask：Bernoulli 采样，将部分权重置 0
                    mask = torch.bernoulli(
                        torch.full_like(p, 1.0 - ratio)
                    )
                    p.data.mul_(mask)

                elif self.mode == "noise":
                    # 噪声注入：加高斯噪声（模拟"不精确"的信号传输）
                    noise = torch.randn_like(p) * self.noise_std * ratio
                    p.data.add_(noise)

                elif self.mode == "struct":
                    # 结构化 mask：按输出通道整体 mask
                    if p.dim() == 4:  # conv weight: [out_c, in_c, k, k]
                        out_c = p.size(0)
                        mask = torch.bernoulli(
                            torch.full((out_c, 1, 1, 1), 1.0 - ratio, device=p.device)
                        )
                        p.data.mul_(mask.expand_as(p))
                    elif p.dim() == 2:  # linear weight: [out, in]
                        out = p.size(0)
                        mask = torch.bernoulli(
                            torch.full((out, 1), 1.0 - ratio, device=p.device)
                        )
                        p.data.mul_(mask.expand_as(p))

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
    mask_hook: Optional[WeightMaskHook] = None,
    log_interval: int = 50,
    epoch: int = 0,
) -> Tuple[float, float]:
    """训练一个 epoch，返回 (avg_loss, accuracy)。"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # 关键：在 optimizer step 后施加 mask
        if mask_hook is not None:
            mask_hook.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % log_interval == 0:
            print(f"  [{epoch}][{batch_idx:3d}/{len(loader):3d}] "
                  f"loss: {loss.item():.4f} "
                  f"acc: {100. * correct / total:.2f}%"
                  + (f" mask_ratio: {mask_hook.get_current_mask_ratio():.3f}"
                     if mask_hook else ""))

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
    model = resnet18(num_classes=10, in_channels=config.get("in_channels", 3)).to(device)
    print(f"  模型参数量: {count_model_params(model):,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.get("lr", 0.1),
        momentum=config.get("momentum", 0.9),
        weight_decay=config.get("weight_decay", 5e-4),
        nesterov=True,
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-4
    )

    # 初始化 mask hook
    mask_hook = None
    if config["mask_mode"] != "none":
        mask_hook = WeightMaskHook(
            model=model,
            mode=config["mask_mode"],
            mask_ratio=config.get("mask_ratio", 0.5),
            total_epochs=config["epochs"],
            mask_epochs=config.get("mask_epochs", config["epochs"] // 2),
            schedule=config.get("mask_schedule", "step"),
            noise_std=config.get("noise_std", 0.01),
        )

    # 训练日志
    log = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "mask_ratio": [],
        "lr": [],
        "config": config,
    }

    best_acc = 0.0
    best_state = None

    for epoch in range(1, config["epochs"] + 1):
        if mask_hook is not None:
            mask_hook.set_epoch(epoch)

        print(f"\n--- Epoch {epoch}/{config['epochs']} "
              f"| lr: {scheduler.get_last_lr()[0]:.6f} "
              + (f"| mask_ratio: {mask_hook.get_current_mask_ratio() if mask_hook else 0:.3f}"))

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            mask_hook=mask_hook, epoch=epoch,
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        log["train_loss"].append(train_loss)
        log["train_acc"].append(train_acc)
        log["test_loss"].append(test_loss)
        log["test_acc"].append(test_acc)
        log["mask_ratio"].append(mask_hook.get_current_mask_ratio() if mask_hook else 0)
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
        "hard": "#F44336",
        "noise": "#4CAF50",
        "struct": "#FF9800",
    }
    labels = {
        "none": "Baseline (Normal)",
        "hard": "Hard Mask (Weight Zeroing)",
        "noise": "Noise Injection (Gaussian)",
        "struct": "Structured Mask (Channel-wise)",
    }
    linestyles = {
        "none": "-",
        "hard": "--",
        "noise": ":",
        "struct": "-.",
    }

    for (name, log) in results.items():
        color = colors.get(log["config"]["mask_mode"], "#666")
        ls = linestyles.get(log["config"]["mask_mode"], "-")
        label = labels.get(log["config"]["mask_mode"], name)
        epochs = range(1, len(log["test_acc"]) + 1)

        mask_epochs = log["config"].get("mask_epochs", log["config"]["epochs"] // 2)

        # 在图上画遮罩切换线
        for ax in axes.flat:
            ax.axvline(x=mask_epochs, color="gray", linestyle="--",
                       alpha=0.3, label="_mask_boundary")

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

    # 加一行 mask_ratio 和 lr 图
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    for (name, log) in results.items():
        color = colors.get(log["config"]["mask_mode"], "#666")
        label = labels.get(log["config"]["mask_mode"], name)
        epochs = range(1, len(log["mask_ratio"]) + 1)
        axes2[0].plot(epochs, log["mask_ratio"], color=color, lw=2, label=label)
        axes2[1].plot(epochs, log["lr"], color=color, lw=2, label=label)

    axes2[0].set_title("Mask Ratio (per epoch)", fontsize=13)
    axes2[0].set_xlabel("Epoch")
    axes2[0].legend(fontsize=9)
    axes2[0].grid(True, alpha=0.3)

    axes2[1].set_title("Learning Rate", fontsize=13)
    axes2[1].set_xlabel("Epoch")
    axes2[1].legend(fontsize=9)
    axes2[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_dir / "comparison_accuracy_loss.png", dpi=150, bbox_inches="tight")
    fig2.savefig(save_dir / "mask_lr_schedule.png", dpi=150, bbox_inches="tight")
    print(f"  图片已保存: {save_dir}/comparison_accuracy_loss.png")
    print(f"  图片已保存: {save_dir}/mask_lr_schedule.png")
    plt.close("all")


# ============================================================================
# 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Weight Masking Experiment")
    parser.add_argument("--epochs", type=int, default=200,
                        help="总训练 epoch 数 (默认: 200)")
    parser.add_argument("--mask-epochs", type=int, default=None,
                        help="前多少 epoch 施加 mask (默认: epochs//2)")
    parser.add_argument("--mask-ratio", type=float, default=0.5,
                        help="初始 mask 比例 (默认: 0.5)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="batch size (默认: 128)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="初始学习率 (默认: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子 (默认: 42)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader worker 数 (默认: 2)")
    parser.add_argument("--skip-none", action="store_true",
                        help="跳过 Baseline (none) 实验")
    parser.add_argument("--modes", type=str, nargs="+",
                        choices=["none", "hard", "noise", "struct"],
                        default=["none", "hard", "noise", "struct"],
                        help="运行哪些 mask 模式 (默认: 全部)")
    parser.add_argument("--schedule", type=str, default="step",
                        choices=["step", "linear", "cosine", "inverse"],
                        help="mask 衰减 schedule (默认: step)")
    parser.add_argument("--noise-std", type=float, default=0.01,
                        help="噪声注入标准差 (默认: 0.01)")
    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "fashion_mnist"],
                        help="数据集 (默认: cifar10, fashion_mnist 更快下载)")
    args = parser.parse_args()

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
    mask_epochs = args.mask_epochs if args.mask_epochs else epochs // 2

    base_config = {
        "epochs": epochs,
        "mask_epochs": mask_epochs,
        "mask_ratio": args.mask_ratio,
        "mask_schedule": args.schedule,
        "noise_std": args.noise_std,
        "lr": args.lr,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "dataset": args.dataset,
        "in_channels": in_channels,
    }

    print(f"\n实验配置:")
    print(f"  Epochs: {epochs} | Mask Epochs: {mask_epochs} | Mask Ratio: {args.mask_ratio}")
    print(f"  Schedule: {args.schedule} | Noise Std: {args.noise_std}")
    print(f"  Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  模式: {args.modes}")

    # 计算结果保存路径
    save_dir = Path(f"results_{args.seed}")
    save_dir.mkdir(exist_ok=True)
    print(f"\n结果保存: {save_dir.resolve()}")

    # 运行实验
    results = {}
    for mode in args.modes:
        config = {**base_config, "name": f"mask_{mode}", "mask_mode": mode}
        log = run_experiment(config, device, train_loader, test_loader, save_dir)
        results[f"mask_{mode}"] = log

        # 保存每个实验的日志
        with open(save_dir / f"log_{mode}.json", "w") as f:
            # 把 numpy 值转成 float
            clean = {
                k: (v if not isinstance(v, list) else
                    [float(x) if not isinstance(x, (int, float)) else x for x in v])
                for k, v in log.items() if k != "config"
            }
            clean["config"] = {
                k: v for k, v in log["config"].items()
                if isinstance(v, (str, int, float, bool))
            }
            json.dump(clean, f, indent=2)

    # 画对比图
    draw_comparison(results, save_dir)

    # 打印总结
    print(f"\n{'='*60}")
    print(f"  📊 实验总结")
    print(f"{'='*60}")
    for name, log in results.items():
        label = labels_map.get(log["config"]["mask_mode"], name)
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
