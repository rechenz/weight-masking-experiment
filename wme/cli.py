"""
Weight Masking Experiment CLI
=============================
类比人类婴儿"机能不全→健全"的训练策略验证。

策略：训练前期随机屏蔽部分权重（模拟机能不全），后期放开（模拟机能健全）。
对比正常训练，看对收敛、泛化、创造性的影响。

用法:
    python -m wme
    python -m wme --light
    python -m wme --model resnet18 --modes dropout
"""

from wme import (
    create_model,
    run_experiment,
    draw_comparison,
    _convert_to_json_safe,
)
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.utils.data as data
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Activation Constraint Experiment")
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
                        choices=["step", "linear", "decay",
                                 "cosine", "inverse", "triangle"],
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
        if not args.amp:
            print("  ⚡ --light: 启用 AMP")
            args.amp = True

    # 检查模型与约束模式的兼容性
    if args.model != "vit":
        for mode in args.modes:
            if mode != "none":
                print(f"  ⚠️  注意: {args.model} 的约束效果与 TinyViT 不同。"
                      f"约束将作用于分类器之前的特征层。")

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
    constraint_epochs = args.constraint_epochs if args.constraint_epochs is not None else epochs // 2

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
    }

    print(f"\n实验配置:")
    print(
        f"  Epochs: {epochs} | Constraint Epochs: {constraint_epochs} | Max Rate: {args.max_rate}")
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
    with open(save_dir / "log_none.json", "w", encoding="utf-8") as f:
        json.dump(_convert_to_json_safe(log), f, indent=2, ensure_ascii=False)

    # 再各个 schedule 跑 mask 实验
    for schedule in args.schedules:
        for mode in args.modes:
            if mode == "none":
                continue  # baseline 已经跑过了
            name = f"constraint_{mode}_{schedule}"
            cfg = {**base_config, "name": name, "constraint_mode": mode,
                   "constraint_schedule": schedule}
            log = run_experiment(cfg, device, train_loader,
                                 test_loader, save_dir)
            results[name] = log
            with open(save_dir / f"log_{mode}_{schedule}.json", "w", encoding="utf-8") as f:
                json.dump(_convert_to_json_safe(log), f,
                          indent=2, ensure_ascii=False)

    # 画对比图 — 每个 schedule 一张
    schedules_to_plot = ["none"] + \
        args.schedules if args.schedules != ["none"] else ["none"]
    for schedule in schedules_to_plot:
        subset = {k: v for k, v in results.items()
                  if schedule == "none" and "none" in k
                  or schedule in k}
        if len(subset) > 1:
            draw_comparison(subset, save_dir, schedule_tag=schedule)

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


if __name__ == "__main__":
    main()
