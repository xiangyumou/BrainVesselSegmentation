from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class DevicePolicy:
    device: torch.device
    amp_enabled: bool


def select_device(requested: str = "auto") -> DevicePolicy:
    requested = requested.lower()
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported device '{requested}'; choose auto, cuda, mps, or cpu")

    if requested in {"auto", "cuda"} and torch.cuda.is_available():
        return DevicePolicy(torch.device("cuda"), True)
    if requested == "cuda":
        raise RuntimeError("CUDA was requested but is not available")

    mps_available = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )
    if requested in {"auto", "mps"} and mps_available:
        return DevicePolicy(torch.device("mps"), False)
    if requested == "mps":
        raise RuntimeError("MPS was requested but is not available")
    return DevicePolicy(torch.device("cpu"), False)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

