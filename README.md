# Weight Masking Experiment 🧪

> 给神经网络戴沙袋训练，看卸掉之后能不能飞起来。

**灵感来源：** 人类婴儿出生时身体机能不全（感官、运动控制等），随着年龄增长逐渐健全。这本质上是一个「约束 → 放开」的过程——是不是跟降噪模型 / 稀疏训练有异曲同工之处？

如果把这种思路用在模型训练上：**前期给权重加约束（随机屏蔽 / 加噪声），后期放开**，会怎样？

## 实验设计

### 模型
- ResNet-18 (CIFAR-10 版, ~11M 参数)
- 数据集: CIFAR-10 (50K 训练 / 10K 测试)

### 对比策略

| 模式 | 描述 | 类比 |
|------|------|------|
| **Baseline** (none) | 正常训练，不施加任何约束 | 正常发育 |
| **硬 Mask** (hard) | 训练前期随机将部分权重置零 | 部分神经连接受阻 |
| **噪声注入** (noise) | 训练前期给权重加高斯噪声 | 信号传输不精确 |
| **结构化 Mask** (struct) | 按输出通道整体 mask（整组关闭） | 某脑区发育不全 |

### 调度策略 (Schedule)
- **step**: 前 N 个 epoch 固定 mask 比例，之后全部放开
- **linear**: mask 比例线性递减
- **cosine**: 余弦衰减
- **inverse**: 前期大比例，快速衰减

### 训练参数
- 优化器: SGD + Nesterov momentum 0.9, weight decay 5e-4
- 学习率: 0.1, CosineAnnealing 至 1e-4
- Batch size: 128
- Epochs: 200 (默认前 100 mask，后 100 放开)

## 快速开始

```bash
# 克隆
git clone https://github.com/rechenz/weight-masking-experiment.git
cd weight-masking-experiment

# 跑完整实验（全部4种模式对比）
python experiment.py

# 自定义配置
python experiment.py --epochs 100 --mask-ratio 0.3 --schedule cosine
python experiment.py --modes none hard noise    # 只跑特定模式
python experiment.py --mask-epochs 50           # 前期只mask 50个epoch
```

## 结果

实验结果保存在 `results_<seed>/` 目录下：
- `comparison_accuracy_loss.png` — 四张图对比 train/test acc & loss
- `mask_lr_schedule.png` — mask ratio 和 lr 变化曲线
- `log_<mode>.json` — 每个实验的完整日志
- `best_<mode>.pt` — 最佳 checkpoint

## 可能的观察

1. **硬 Mask 可能在前中期落后，但放开后追击** — 如果观察到测试集上收敛曲线在切换点后斜率显著增大，说明约束期确实学到了某种鲁棒表征
2. **结构化 Mask 可能最有意思** — 整组关闭某些通道类似"特定脑区不发育"，放开后可能产生独特的表征组合
3. **噪声注入可能最稳定** — 比硬 mask 更平滑，训练不容易崩，放开后泛化能力可能最强
4. **如果所有 mask 策略都显著差于 baseline** — 说明在权重层面加约束可能太难了，要试输入层面或激活值层面

## 延伸想法

如果这个实验能验证「约束前训练 → 放开微调 → 更好泛化」的趋势，下一步可以：
1. 在语言模型上试（小 GPT，权重 mask 一些 attention head）→ 看创造力指标
2. 扩展到具身智能场景 — 前期限制传感器的精度或响应范围
3. 和知识蒸馏结合 — 约束后的子网络在放开后表现如何

---

> 写于一个下午。如果人类婴儿能长大，那神经网络也行。
