from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Sequence

import torch
import wandb

from fidelity.foreign.loss import estimate_loss_coefficients, get_vaeloss
from fidelity.foreign.utils import set_gmm_centers


class LossFunc(ABC):
    @abstractmethod
    def __call__(self, z: Sequence[Any], y: Sequence[Any]) -> Iterable[torch.Tensor]:
        ...


class LambdaLossFunc(LossFunc):
    def __init__(self, lambda_: Callable[[Sequence[Any], Sequence[Any]], Iterable[torch.Tensor]]) -> None:
        super().__init__()
        self.__lambda = lambda_

    def __call__(self, z: Sequence[Any], y: Sequence[Any]) -> Iterable[torch.Tensor]:
        return self.__lambda(z, y)


class ForeignVAELoss(LossFunc):
    def __init__(self) -> None:
        super().__init__()
        self.__gmm_centers, self.__gmm_std = set_gmm_centers(wandb.config["training"]["loss_fn"]["latent_dim"], wandb.config["training"]["loss_fn"]["num_clusters"])
        self.__ks_weight, self.__cv_weight = estimate_loss_coefficients(wandb.config["training"]["nograd_batch_size"], self.__gmm_centers, self.__gmm_std, num_samples=64)

    def __call__(self, z: Sequence[Any], y: Sequence[Any]) -> Iterable[torch.Tensor]:
        return get_vaeloss(z[0], z[1], y[0], self.__ks_weight, self.__cv_weight, wandb.config["training"]["loss_fn"]["img_loss_weight"], self.__gmm_centers, self.__gmm_std)[:1]


def build_loss_fn() -> LossFunc:
    match wandb.config["training"]["loss_fn"]["name"]:
        case "MSE":
            return LambdaLossFunc(lambda z, y: [torch.nn.functional.mse_loss(z[0], y[0])])
        case "ForeignVAELoss":
            return ForeignVAELoss()
        case _ as fn:
            raise ValueError(f"Unknown loss function {fn}")
