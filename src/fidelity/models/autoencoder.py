from __future__ import annotations

from typing import Tuple

import torch


class Reshape(torch.nn.Module):
    def __init__(self, out_shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.__out_shape = out_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, *self.__out_shape)


class Autoencoder(torch.nn.Module):
    def __init__(self, latent_size: int) -> None:
        super().__init__()
        self.__predicting = False
        self.__encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 2, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(2),
            torch.nn.AvgPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(2, 4, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.AvgPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(4, 8, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.AvgPool2d(2, ceil_mode=True),
            torch.nn.Conv2d(8, 16, kernel_size=3, padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Flatten(),
            torch.nn.Linear(256, latent_size),
        )
        self.__decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_size, 256),
            Reshape((16, 4, 4)),
            torch.nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(8),
            torch.nn.ConvTranspose2d(8, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.PReLU(),
            torch.nn.BatchNorm2d(4),
            torch.nn.ConvTranspose2d(4, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        )

    def train(self, mode: bool = True) -> Autoencoder:
        self.__predicting = False
        return super().train(mode)

    def eval(self) -> Autoencoder:
        self.__predicting = False
        return super().eval()

    def pred(self) -> Autoencoder:
        super().eval()
        self.__predicting = True
        return self

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.__predicting:
            lat = self.__encoder(x)
        else:
            lat = x
        return self.__decoder(lat), lat
