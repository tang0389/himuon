import os
import random
import warnings
import torch
import numpy as np

import time
from datetime import timedelta


def seed_everything(SEED: int):
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_kv_args(raw: list[str]) -> dict:
    """Parse ['beta_ema=0.95', 'lanczos_steps=10'] into {'beta_ema': 0.95, 'lanczos_steps': 10}."""
    import ast

    result = {}
    for kv in raw:
        k, v = kv.split("=", 1)
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            pass  # keep as string
        result[k] = v
    return result


def merge_kv_args(args, raw: list[str] | None = None):
    """Parse key=val pairs and merge into args namespace. Warns on overriding existing non-None attributes."""
    if raw is None:
        raw = getattr(args, "kwargs", [])
    for k, v in parse_kv_args(raw).items():
        if hasattr(args, k) and getattr(args, k) is not None:
            warnings.warn(
                f"'--kwargs {k}={v}' overrides existing arg '{k}={getattr(args, k)}'"
            )
        setattr(args, k, v)
    return args


class TimeBudget:
    def __init__(self, hours: float | None = None):
        self.start_time = time.monotonic()
        self.deadline = self.start_time + (hours * 3600) if hours else float("inf")

    def is_expired(self) -> bool:
        return time.monotonic() >= self.deadline

    def __repr__(self):
        remaining = max(0, self.deadline - time.monotonic())
        return str(timedelta(seconds=int(remaining)))  # format: "HH:MM:SS"
