from typing import cast

import torch
import wandb


class DummyLrScheduler(torch.optim.lr_scheduler._LRScheduler):  # type: ignore
    def get_lr(self) -> float:
        return self.base_lrs  # type: ignore


def build_lr_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler._LRScheduler:  # type: ignore
    if "lr_scheduler" in wandb.config["training"]:
        return DummyLrScheduler(optimizer)
    match wandb.config["training"]["lr_scheduler"]["name"]:
        case "ReduceLROnPlateau":
            return cast(torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer))  # type: ignore
        case _ as lrs:
            raise ValueError(f"Unknown learning rate scheduler {lrs}")
