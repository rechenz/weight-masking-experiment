"""
模型定义模块
=============
- ResNet-18 (CIFAR-10 适配版)
- SmallCNN (轻量 CNN)
- TinyViT (超轻量 ViT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Protocol, runtime_checkable


@runtime_checkable
class ConstraintModel(Protocol):
    """约束模型接口协议。
    所有支持激活层约束的模型必须实现这两个方法。
    """

    def set_dropout_rate(self, rate: float) -> None: ...
    def set_actnoise_std(self, std: float) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


# ============================================================================
# ResNet-18 for CIFAR-10
# ============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet-18 for CIFAR-10，带激活层约束支持。"""

    def __init__(self, block: type, num_blocks: list, num_classes: int = 10,
                 in_channels: int = 3) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # 激活层约束参数（运行时动态调整）
        self.forward_dropout_rate: float = 0.0
        self.actnoise_std: float = 0.0

    def _make_layer(self, block: type, planes: int, num_blocks: int,
                    stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def set_dropout_rate(self, rate: float) -> None:
        """动态调整 dropout 率。"""
        self.forward_dropout_rate = rate

    def set_actnoise_std(self, std: float) -> None:
        """动态调整激活噪声幅度。"""
        self.actnoise_std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=self.forward_dropout_rate,
                        training=self.training)
        if self.actnoise_std > 0 and self.training:
            out = out + torch.randn_like(out) * self.actnoise_std
        out = self.linear(out)
        return out


def resnet18(num_classes: int = 10, in_channels: int = 3) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, in_channels)


# ============================================================================
# SmallCNN — 轻量 CNN
# ============================================================================

class SmallCNN(nn.Module):
    """轻量 CNN，~250K 参数，GPU 负载只有 ResNet-18 的 1/40。"""

    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(
                32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(
                64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(
                128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Linear(256, num_classes)
        # 激活层约束参数（运行时动态调整）
        self.forward_dropout_rate: float = 0.0
        self.actnoise_std: float = 0.0

    def set_dropout_rate(self, rate: float) -> None:
        """动态调整 dropout 率。"""
        self.forward_dropout_rate = rate

    def set_actnoise_std(self, std: float) -> None:
        """动态调整激活噪声幅度。"""
        self.actnoise_std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = F.dropout(x, p=self.forward_dropout_rate, training=self.training)
        if self.actnoise_std > 0 and self.training:
            x = x + torch.randn_like(x) * self.actnoise_std
        x = self.classifier(x)
        return x


# ============================================================================
# TinyViT — 超轻量 Transformer
# ============================================================================

class PatchEmbed(nn.Module):
    """图像分块 + 线性投影。"""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 1, embed_dim: int = 192) -> None:
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )
        self.forward_dropout_rate: float = 0.0  # 运行时动态调整

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + F.dropout(self.mlp(self.norm2(x)),
                          p=self.forward_dropout_rate, training=self.training)
        return x


class TinyViT(nn.Module):
    """超轻量 ViT，适合 32x32 小图，~2.5M 参数。"""

    def __init__(self, img_size: int = 32, patch_size: int = 4,
                 in_channels: int = 1, num_classes: int = 10,
                 embed_dim: int = 192, depth: int = 6,
                 num_heads: int = 6, mlp_ratio: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.actnoise_std: float = 0.0  # 运行时动态调整

        self._init_weights()

    def set_dropout_rate(self, rate: float) -> None:
        """动态调整所有 block 的 dropout 率。"""
        for block in self.blocks:
            setattr(block, 'forward_dropout_rate', rate)

    def set_actnoise_std(self, std: float) -> None:
        """动态调整激活噪声幅度。"""
        self.actnoise_std = std

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0]  # CLS token
        if self.actnoise_std > 0 and self.training:
            x = x + torch.randn_like(x) * self.actnoise_std
        x = self.head(x)
        return x


# ============================================================================
# 工厂函数
# ============================================================================

def create_model(name: str, num_classes: int,
                 in_channels: int) -> nn.Module:
    if name == "resnet18":
        return resnet18(num_classes, in_channels)
    elif name == "smallcnn":
        return SmallCNN(in_channels, num_classes)
    elif name == "vit":
        return TinyViT(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
