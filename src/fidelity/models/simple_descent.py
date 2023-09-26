from __future__ import annotations

import torch
import wandb

from fidelity.foreign.loss import estimate_loss_coefficients, get_vaeloss
from fidelity.foreign.utils import set_gmm_centers


class SimpleDescent(torch.nn.Module):
    def __init__(self, iterations: int, lr: float, img_weight: float) -> None:
        super().__init__()
        self.__iterations = iterations
        self.__lr = lr
        self.__gmm_centers, self.__gmm_std = set_gmm_centers(1, 1)
        self.__ks_weight, self.__cv_weight = estimate_loss_coefficients(wandb.config["training"]["nograd_batch_size"], self.__gmm_centers, self.__gmm_std, num_samples=64)
        self.__img_weight = img_weight

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        u_hat = torch.zeros_like(f, requires_grad=True)
        optimizer = torch.optim.Adam([u_hat], lr=self.__lr)
        for _ in range(self.__iterations):
            optimizer.zero_grad()
            with torch.enable_grad():
                loss = get_vaeloss(u_hat, torch.pow(u_hat - f, 2).reshape(f.shape[0], -1), f, self.__ks_weight, self.__cv_weight, self.__img_weight, self.__gmm_centers, self.__gmm_std)[0]
                loss.backward()
            optimizer.step()
        return u_hat
