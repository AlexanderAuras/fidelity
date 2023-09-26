from pathlib import Path
from typing import Any, Dict, Literal, Optional, Sized, Tuple, Union, cast

import torch.utils.data
import torchvision
import wandb

from fidelity.data.feature_mod_dataset import FeatureModificationDataset
from fidelity.data.fixed_noise_dataset import FixedNoiseDataset
from fidelity.data.noise import AWGN
from fidelity.data.random_vec_dataset import RandomVecDataset
from fidelity.data.transform_dataset import TransformDataset


_GENERATOR_STATES: Dict[str, torch.Tensor] = {}
_DATASET_DIR = Path("datasets")


def _reproducible_random_split(dataset_name: str, dataset: torch.utils.data.Dataset[Any], split: float) -> Tuple[torch.utils.data.Dataset[Any], torch.utils.data.Dataset[Any]]:
    generator = torch.Generator()
    if dataset_name in _GENERATOR_STATES:
        generator.set_state(_GENERATOR_STATES[dataset_name])
    else:
        _GENERATOR_STATES[dataset_name] = generator.get_state()
    len1 = int(len(cast(Sized, dataset)) * split)
    len2 = len(cast(Sized, dataset)) - int(len(cast(Sized, dataset)) * split)
    return cast(Tuple[torch.utils.data.Dataset[Any], torch.utils.data.Dataset[Any]], torch.utils.data.random_split(dataset, [len1, len2], generator))


def load_dataset(subset: Union[Literal["train"], Literal["val"], Literal["test"], Literal["pred"]], idx: Optional[int] = None) -> torch.utils.data.Dataset[Tuple[Any, Any]]:
    if subset == "train":
        dataset_config = wandb.config["data"][f"train_dataset"]
    else:
        dataset_config = wandb.config["data"][f"{subset}_datasets"][idx]

    match dataset_config["name"]:
        case "MNIST":
            match subset:
                case "train":
                    dataset = torchvision.datasets.MNIST(str(_DATASET_DIR.resolve()), train=True, download=True)
                    dataset = _reproducible_random_split(dataset_config["name"], dataset, dataset_config["train_split"])[0]
                case "val":
                    dataset = torchvision.datasets.MNIST(str(_DATASET_DIR.resolve()), train=True, download=True)
                    dataset = _reproducible_random_split(dataset_config["name"], dataset, dataset_config["train_split"])[1]
                case "test":
                    dataset = torchvision.datasets.MNIST(str(_DATASET_DIR.resolve()), train=False, download=True)
                case "pred":
                    dataset = torchvision.datasets.MNIST(str(_DATASET_DIR.resolve()), train=False, download=True)
        case "Kuzushiji":
            match subset:
                case "train":
                    dataset = torchvision.datasets.KMNIST(str(_DATASET_DIR.resolve()), train=True, download=True)
                    dataset = _reproducible_random_split(dataset_config["name"], dataset, dataset_config["train_split"])[0]
                case "val":
                    dataset = torchvision.datasets.KMNIST(str(_DATASET_DIR.resolve()), train=True, download=True)
                    dataset = _reproducible_random_split(dataset_config["name"], dataset, dataset_config["train_split"])[1]
                case "test":
                    dataset = torchvision.datasets.KMNIST(str(_DATASET_DIR.resolve()), train=False, download=True)
                case "pred":
                    dataset = torchvision.datasets.KMNIST(str(_DATASET_DIR.resolve()), train=False, download=True)
        case "FashionMNIST":
            match subset:
                case "train":
                    dataset = torchvision.datasets.FashionMNIST(str(_DATASET_DIR.resolve()), train=True, download=True)
                    dataset = _reproducible_random_split(dataset_config["name"], dataset, dataset_config["train_split"])[0]
                case "val":
                    dataset = torchvision.datasets.FashionMNIST(str(_DATASET_DIR.resolve()), train=True, download=True)
                    dataset = _reproducible_random_split(dataset_config["name"], dataset, dataset_config["train_split"])[1]
                case "test":
                    dataset = torchvision.datasets.FashionMNIST(str(_DATASET_DIR.resolve()), train=False, download=True)
                case "pred":
                    dataset = torchvision.datasets.FashionMNIST(str(_DATASET_DIR.resolve()), train=False, download=True)
        case "RandomVec":
            match subset:
                case "train":
                    raise RuntimeError("RandomVec datasets have no ground_truth and cannot be used as training datasets")
                case "val":
                    raise RuntimeError("RandomVec datasets have no ground_truth and cannot be used as validation datasets")
                case "test":
                    raise RuntimeError("RandomVec datasets have no ground_truth and cannot be used as test datasets")
                case "pred":
                    return RandomVecDataset(dataset_config["shape"], dataset_config["count"])
        case _ as ds:
            raise ValueError(f"Unknown dataset {ds}")

    unified_dataset = FeatureModificationDataset(dataset, 0, 0)
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(lambda x: torchvision.transforms.Grayscale()(x[0:3]) if x.shape[0] == 4 else (torchvision.transforms.Grayscale()(x) if x.shape[0] == 3 else x[0:1])),
            torchvision.transforms.Resize(wandb.config["data"]["img_size"], antialias=cast(str, True)),
            torchvision.transforms.CenterCrop(wandb.config["data"]["img_size"]),
        ]
    )
    transform_dataset = TransformDataset(unified_dataset, {0: transform}, {0: transform})
    noisy_dataset = FixedNoiseDataset(transform_dataset, AWGN(wandb.config["data"]["noise_level"]), input_=(0,))
    return noisy_dataset
