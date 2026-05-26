"""Deterministic seeding utility.

Per-module seeds are derived from a single root seed using SHA-256 of the
module name, ensuring reproducibility while keeping per-module random
streams independent.
"""

from __future__ import annotations

import hashlib
import random
from typing import Optional

import numpy as np


def derive_seed(root_seed: int, module_name: str) -> int:
    """Return a deterministic int32 seed for a given module."""
    payload = str(root_seed).encode("utf-8") + b"::" + module_name.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return int(digest, 16)


def set_global_seeds(seed: int, also_torch: bool = True) -> None:
    """Seed Python's random, NumPy, and (optionally) PyTorch CPU RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    if also_torch:
        try:
            import torch

            torch.manual_seed(seed)
            if hasattr(torch, "use_deterministic_algorithms"):
                # Best-effort; some ops have no deterministic implementation.
                try:
                    torch.use_deterministic_algorithms(False)
                except Exception:
                    pass
        except ImportError:
            pass


def module_rng(root_seed: int, module_name: str) -> np.random.Generator:
    """Return a NumPy Generator seeded per module."""
    return np.random.default_rng(derive_seed(root_seed, module_name))
