"""
工具函数模块
===========
"""

import numpy as np
import torch


def _convert_to_json_safe(obj):
    """递归将对象转为 JSON 安全的 Python 原生类型。"""
    if isinstance(obj, torch.Tensor):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [_convert_to_json_safe(v) for v in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)
