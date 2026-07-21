"""
主实验与绘图模块
===============
"""

from copy import deepcopy
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from wme.models import create_model
from wme.constraint import ActivationConstraint
from wme.training import train_epoch, evaluate, count_model_params


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
                optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, total_iters=10),
                optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, config["epochs"] - 10), eta_min=1e-5),
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
            constraint_epochs=config.get(
                "constraint_epochs", config["epochs"] // 2),
            schedule=config.get("constraint_schedule", "decay"),
        )

    use_amp = config.get("amp", False)
    scaler = torch.amp.GradScaler(device.type) if use_amp and device.type != "cpu" else None

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
    # 初始化变量以防止 epochs=0 时在循环外未绑定
    test_loss = 0.0
    test_acc = 0.0

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
        log["constraint_rate"].append(
            constraint.get_current_rate() if constraint else 0)
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
    print(
        f"\n  🔚 最终结果: test_acc={final_acc:.2f}%  |  best_acc={best_acc:.2f}%")

    log["final_test_acc"] = final_acc
    log["best_test_acc"] = best_acc

    return log


def draw_comparison(results: dict, save_dir: Path, schedule_tag: str = "all") -> None:
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

    # 先收集所有结果的约束切换线位置，只画一次
    constraint_boundaries = set()
    for log in results.values():
        ce = log["config"].get("constraint_epochs",
                               log["config"]["epochs"] // 2)
        constraint_boundaries.add(ce)

    for ce in constraint_boundaries:
        for ax in axes.flat:
            ax.axvline(x=ce, color="gray", linestyle="--",
                       alpha=0.3, label="_constraint_boundary")

    for (_name, log) in results.items():
        cmode = log["config"].get("constraint_mode", "none")
        color = colors.get(cmode, "#666")
        ls = linestyles.get(cmode, "-")
        label: str = labels.get(cmode, _name) or _name
        sched = log["config"].get("constraint_schedule", "")
        if sched and sched != "none":
            label += f" ({sched})"
        epochs = range(1, len(log["test_acc"]) + 1)

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
    for (_name, log) in results.items():
        cmode = log["config"].get("constraint_mode", "none")
        color = colors.get(cmode, "#666")
        sched = log["config"].get("constraint_schedule", "")
        label: str = labels.get(cmode, _name) or _name
        if sched and sched != "none":
            label += f" ({sched})"
        epochs = range(1, len(log["constraint_rate"]) + 1)
        axes2[0].plot(epochs, log["constraint_rate"],
                      color=color, lw=2, label=label)
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
    fig.savefig(save_dir / f"comparison_accuracy_loss_{schedule_tag}.png",
                dpi=150, bbox_inches="tight")
    fig2.savefig(save_dir / f"constraint_lr_schedule_{schedule_tag}.png",
                 dpi=150, bbox_inches="tight")
    print(f"  图片已保存: {save_dir}/comparison_accuracy_loss.png")
    print(f"  图片已保存: {save_dir}/constraint_lr_schedule.png")
    plt.close("all")
