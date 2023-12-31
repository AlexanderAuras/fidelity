import torch
import wandb

from fidelity.foreign.model import VAE as ForeignVAE
from fidelity.models.autoencoder import Autoencoder
from fidelity.models.simple_descent import SimpleDescent


def build_model() -> torch.nn.Module:
    match wandb.config["model"]["name"]:
        case "Autoencoder":
            return Autoencoder(latent_size=wandb.config["model"]["latent_size"])
        case "ForeignVAE":
            return ForeignVAE(
                dataset="MNIST",
                nc=1,
                ndf=wandb.config["model"]["ndf"],
                nef=wandb.config["model"]["nef"],
                nz=wandb.config["model"]["latent_size"],
                isize=wandb.config["data"]["img_size"][0],
                latent_noise_scale=wandb.config["model"]["latent_noise_delta"],
            )
        case "SimpleDescent":
            return SimpleDescent(
                wandb.config["model"]["iterations"],
                10.0 ** wandb.config["model"]["lr_exp"],
                10.0 ** wandb.config["model"]["img_weight_exp"],
                wandb.config["model"]["regularizer"],
                10.0 ** wandb.config["model"]["reg_weight_exp"],
                10.0 ** wandb.config["model"]["ks_weight_exp"],
                10.0 ** wandb.config["model"]["cv_weight_exp"],
            )
        case _ as mdl:
            raise ValueError(f'Unknown model "{mdl}"')
