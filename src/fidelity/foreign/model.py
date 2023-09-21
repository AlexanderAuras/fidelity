""" model
    This file contains VAE model definition for MNIST, FASHIONMNIST, SVHN and CELEBA dataset.
"""
from typing import Literal, Tuple, Union

import torch
import torch.utils.data
from torch import nn


class MNISTEncoder(nn.Module):
    def __init__(self, nc: int, nef: int, nz: int, isize: int, device: torch.device) -> None:
        super(MNISTEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)
        self.encoder = nn.Sequential(nn.Conv2d(nc, nef, 4, 2, padding=1), nn.BatchNorm2d(nef), nn.ReLU(True), nn.Conv2d(nef, nef * 2, 4, 2, padding=1), nn.BatchNorm2d(nef * 2), nn.ReLU(True), nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1), nn.BatchNorm2d(nef * 4), nn.ReLU(True), nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1), nn.BatchNorm2d(nef * 8), nn.ReLU(True))
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        # Reshape
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class MNISTDecoder(nn.Module):
    def __init__(self, nc: int, ndf: int, nz: int, isize: int) -> None:
        super(MNISTDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(
            nn.Linear(nz, 2 * 2 * 1024),
            nn.ReLU(True),
        )
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.decoder_conv = nn.Sequential(nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf * 4), nn.ReLU(True), nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf * 2), nn.ReLU(True), nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(ndf), nn.ReLU(True), nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1), nn.Sigmoid())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 2, 2)
        output = self.decoder_conv(input)
        return output


class SVHNEncoder(nn.Module):
    def __init__(self, nc: int, nef: int, nz: int, isize: int, device: torch.device) -> None:
        super(SVHNEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)
        self.encoder = nn.Sequential(nn.Conv2d(nc, nef, 4, 2, padding=1), nn.BatchNorm2d(nef), nn.ReLU(True), nn.Conv2d(nef, nef * 2, 4, 2, padding=1), nn.BatchNorm2d(nef * 2), nn.ReLU(True), nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1), nn.BatchNorm2d(nef * 4), nn.ReLU(True), nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1), nn.BatchNorm2d(nef * 8), nn.ReLU(True))
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class SVHNDecoder(nn.Module):
    def __init__(self, nc: int, ndf: int, nz: int, isize: int) -> None:
        super(SVHNDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(nn.Linear(nz, 2 * 2 * 1024), nn.ReLU(True))
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.Sequential(self.conv1, nn.BatchNorm2d(ndf * 4), nn.ReLU(True), self.conv2, nn.BatchNorm2d(ndf * 2), nn.ReLU(True), self.conv3, nn.BatchNorm2d(ndf), nn.ReLU(True), self.conv4, nn.Sigmoid())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 2, 2)
        output = self.decoder_conv(input)
        return output


class CELEBEncoder(nn.Module):
    def __init__(self, nc: int, nef: int, nz: int, isize: int, device: torch.device) -> None:
        super(CELEBEncoder, self).__init__()

        # Device
        self.device = device

        # Encoder: (nc, isize, isize) -> (nef*8, isize//8, isize//8)

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, nef, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef),
            nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 2),
            nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 4),
            nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(nef * 8),
        )
        out_size = isize // 16
        self.fc1 = nn.Linear(nef * 8 * out_size * out_size, nz)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Batch size
        batch_size = inputs.size(0)
        hidden = self.encoder(inputs)
        hidden = hidden.view(batch_size, -1)
        latent_z = self.fc1(hidden)
        return latent_z


class CELEBDecoder(nn.Module):
    def __init__(self, nc: int, ndf: int, nz: int, isize: int) -> None:
        super(CELEBDecoder, self).__init__()

        # Map the latent vector to the feature map space
        self.ndf = ndf
        self.out_size = isize // 16
        self.fc1 = nn.Sequential(nn.Linear(nz, 4 * 4 * 1024), nn.ReLU(True))
        # Decoder: (ndf*8, isize//16, isize//16) -> (nc, isize, isize)
        self.conv1 = nn.ConvTranspose2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(ndf * 4, ndf * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(ndf * 2, ndf, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(ndf, nc, kernel_size=4, stride=2, padding=1)
        self.decoder_conv = nn.Sequential(self.conv1, nn.ReLU(True), nn.BatchNorm2d(ndf * 4), self.conv2, nn.ReLU(True), nn.BatchNorm2d(ndf * 2), self.conv3, nn.ReLU(True), nn.BatchNorm2d(ndf), self.conv4, nn.Sigmoid())

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = self.fc1(input)
        input = input.view(input.size(0), 1024, 4, 4)
        output = self.decoder_conv(input)
        return output


class VAE(nn.Module):
    def __init__(self, dataset: Union[Literal["MNIST"], Literal["FASHIONMNIST"], Literal["SVHN"], Literal["CELEB"]] = "MNIST", nc: int = 1, ndf: int = 32, nef: int = 32, nz: int = 16, isize: int = 128, latent_noise_scale: float = 1e-3, device: torch.device = torch.device("cuda:0"), is_train: bool = True) -> None:  # type: ignore
        super(VAE, self).__init__()
        self.nz = nz
        self.is_train = is_train
        self.latent_noise_scale = latent_noise_scale

        if dataset == "MNIST" or dataset == "FASHIONMNIST":
            # Encoder
            self.encoder = MNISTEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = MNISTDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)
        elif dataset == "SVHN":
            # Encoder
            self.encoder = SVHNEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = SVHNDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)
        elif dataset == "CELEB":
            # Encoder
            self.encoder = CELEBEncoder(nc=nc, nef=nef, nz=nz, isize=isize, device=device)
            # Decoder
            self.decoder = CELEBDecoder(nc=nc, ndf=ndf, nz=nz, isize=isize)

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(images)
        if self.is_train:
            z_noise = self.latent_noise_scale * torch.randn((images.size(0), self.nz), device=z.device)
        else:
            z_noise = 0.0
        return self.decode(z + z_noise), z

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
