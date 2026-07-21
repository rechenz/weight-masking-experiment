"""
训练与评估模块
=============
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from wme.constraint import ActivationConstraint


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
            assert scaler is not None
            # type: ignore[attr-defined]
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()  # type: ignore[union-attr]
            scaler.step(optimizer)  # type: ignore[union-attr]
            scaler.update()  # type: ignore[union-attr]
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
