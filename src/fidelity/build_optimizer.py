import torch
import wandb


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    match wandb.config["training"]["optimizer"]["name"]:
        case "Adam":
            return torch.optim.Adam(model.parameters(), lr=wandb.config["training"]["optimizer"]["lr"])
        case _ as opt:
            raise ValueError(f"Unknown optimizer {opt}")
