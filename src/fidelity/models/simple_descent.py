from __future__ import annotations

import torch
import wandb

from fidelity.foreign.loss import estimate_loss_coefficients, get_vaeloss


class SimpleDescent(torch.nn.Module):
    def __init__(self, iterations: int, lr: float, img_weight: float, regularizer: str, reg_weight: float) -> None:
        super().__init__()
        self.__iterations = iterations
        self.__lr = lr
        self.__gmm_centers, self.__gmm_std = torch.tensor([[0.0]], device=wandb.config["device"]), 1
        self.__ks_weight, self.__cv_weight = estimate_loss_coefficients(wandb.config["training"]["grad_batch_size"], self.__gmm_centers, self.__gmm_std, num_samples=64)
        self.__img_weight = img_weight
        match regularizer:
            case "none":
                self.__regularizer = lambda x: 0.0
            case "TV-iso":
                eps = 1e-7
                self.__regularizer = lambda x: (torch.nn.functional.conv2d(x, torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]], [[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=x.dtype, device=x.device), padding=1).pow(2.0).sum(dim=1) + eps).sqrt().mean()
            case "TV-aniso":
                self.__regularizer = lambda x: torch.nn.functional.conv2d(x, torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]], [[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=x.dtype, device=x.device), padding=1).abs().mean()
            case _ as reg:
                raise ValueError(f'Unknown regularizer "{reg}"')
        self.reg_weight = reg_weight

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        u_hat = torch.zeros_like(f, requires_grad=True)
        optimizer = torch.optim.Adam([u_hat], lr=self.__lr)
        for _ in range(self.__iterations):
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = get_vaeloss(u_hat, (u_hat - f).reshape(f.shape[0], -1), f, self.__ks_weight, self.__cv_weight, self.__img_weight, self.__gmm_centers, self.__gmm_std)[0]
                loss += self.reg_weight * self.__regularizer(u_hat)
                loss.backward()
            optimizer.step()
        return u_hat
