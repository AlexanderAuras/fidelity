from __future__ import annotations

from typing import Optional

import torch
import wandb

from fidelity.foreign.loss import estimate_loss_coefficients, get_vaeloss


class SimpleDescent(torch.nn.Module):
    def __init__(
        self,
        iterations: int,
        lr: float,
        img_weight: float,
        regularizer: str,
        reg_weight: float,
        ks_weight: Optional[float],
        cv_weight: Optional[float],
    ) -> None:
        super().__init__()
        self.__iterations = iterations
        self.__lr = lr
        self.__gmm_centers, self.__gmm_std = torch.tensor([[0.0]], device=wandb.config["device"]), wandb.config["data"]["noise_level"]
        if ks_weight is None or cv_weight is None:
            self.__ks_weight, self.__cv_weight = estimate_loss_coefficients(wandb.config["training"]["grad_batch_size"], self.__gmm_centers, self.__gmm_std, num_samples=64)
        else:
            self.__ks_weight = ks_weight
            self.__cv_weight = cv_weight
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

    def forward(self, f: torch.Tensor, print_freq: Optional[int] = None) -> torch.Tensor:
        u_hat = torch.zeros_like(f, requires_grad=True)
        optimizer = torch.optim.Adam([u_hat], lr=self.__lr)
        for i in range(self.__iterations):
            optimizer.zero_grad()
            with torch.enable_grad():
                loss_mean, ks_loss, cov_loss, imgloss = get_vaeloss(u_hat, (u_hat - f).reshape(f.shape[0], -1), f, self.__ks_weight, self.__cv_weight, self.__img_weight, self.__gmm_centers, self.__gmm_std)
                reg_loss = self.reg_weight * self.__regularizer(u_hat)
                if print_freq is not None and i % print_freq == 0:
                    print(f"KS: {ks_loss.item():2.6f}    Cov: {cov_loss.item():2.6f}    Img: {imgloss.item():2.6f}    Reg: {reg_loss.item() if isinstance(reg_loss, torch.Tensor) else reg_loss:2.6f}")
                (loss_mean + reg_loss).backward()
            optimizer.step()
        return u_hat
