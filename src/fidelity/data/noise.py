from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Optional

import torch


class Noise(ABC):
    def __init__(self, level: float) -> None:
        super().__init__()
        self._level = level

    @abstractmethod
    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        ...


class AWGN(Noise):
    def __init__(self, level: float) -> None:
        super().__init__(level)

    def __call__(self, x: torch.Tensor, seed: Optional[int] = None) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(seed if seed is not None else random.randint(0, 999_999_999))
        return x + self._level * torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
