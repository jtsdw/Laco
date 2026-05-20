"""General utilities for LaCo."""

import os
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def find_layers(module, layers=(nn.Conv2d, nn.Linear), name=''):
    """Recursively find layers of given types in a module."""
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers,
            name=name + '.' + name1 if name != '' else name1
        ))
    return res


def format_args_for_logging(args, params_per_line=5):
    """Format argparse.Namespace for readable log output."""
    args_dict = vars(args)
    items = [f"{key}={repr(value)}" for key, value in args_dict.items()]
    lines = []
    for i in range(0, len(items), params_per_line):
        line = ", ".join(items[i:i + params_per_line])
        lines.append(line)
    return "Arguments: " + "\n    ".join(lines)
