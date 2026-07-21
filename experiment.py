#!/usr/bin/env python -u
"""
Weight Masking Experiment
=========================
类比人类婴儿"机能不全→健全"的训练策略验证。

策略：训练前期随机屏蔽部分权重（模拟机能不全），后期放开（模拟机能健全）。
对比正常训练，看对收敛、泛化、创造性的影响。

用法:
    python experiment.py
    python experiment.py --light
    python experiment.py --model resnet18 --modes dropout

（CLI 逻辑已迁移至 wme/cli.py，此文件仅为包外入口包装。）
"""

from wme.cli import main

if __name__ == "__main__":
    main()
