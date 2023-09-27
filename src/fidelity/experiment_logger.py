from typing import Any, Sequence, cast

import torch
import torchmetrics
import torchmetrics.image.fid
import wandb

from fidelity.build_loss_fn import LossFunc


EXAMPLE_COUNT = 8
TENSOR_LOG_COUNT = 5


class ExperimentLogger:
    def __init__(self, loss_fn: LossFunc) -> None:
        super().__init__()
        self.__loss_fn = loss_fn
        self.__loss = 0.0
        self.__mse = 0.0
        self.__psnr = 0.0
        self.__ssim = 0.0
        self.__inputs = []
        self.__targets = []
        self.__outputs = []
        self.__residuals = []
        wandb.define_metric("train.loss", summary="min")
        wandb.define_metric("train.MSE", summary="min")
        wandb.define_metric("train.PSNR", summary="max")
        wandb.define_metric("train.SSIM", summary="max")
        for name in map(lambda x: x["name"], wandb.config["data"]["val_datasets"]):
            wandb.define_metric(f"val.{name}.loss", summary="min")
            wandb.define_metric(f"val.{name}.MSE", summary="min")
            wandb.define_metric(f"val.{name}.PSNR", summary="max")
            wandb.define_metric(f"val.{name}.SSIM", summary="max")
            wandb.define_metric(f"val.{name}.FID", summary="min")
            wandb.define_metric(f"val.{name}.residual_mean")
            wandb.define_metric(f"val.{name}.residual_std")
        for name in map(lambda x: x["name"], wandb.config["data"]["test_datasets"]):
            wandb.define_metric(f"test.{name}.loss", summary="min")
            wandb.define_metric(f"test.{name}.MSE", summary="min")
            wandb.define_metric(f"test.{name}.PSNR", summary="max")
            wandb.define_metric(f"test.{name}.SSIM", summary="max")
            wandb.define_metric(f"test.{name}.FID", summary="min")
            wandb.define_metric(f"test.{name}.residual_mean")
            wandb.define_metric(f"test.{name}.residual_std")
        for name in map(lambda x: x["name"], wandb.config["data"]["pred_datasets"]):
            wandb.define_metric(f"pred.{name}.loss", summary="min")
            wandb.define_metric(f"pred.{name}.MSE", summary="min")
            wandb.define_metric(f"pred.{name}.PSNR", summary="max")
            wandb.define_metric(f"pred.{name}.SSIM", summary="max")
            wandb.define_metric(f"pred.{name}.FID", summary="min")
            wandb.define_metric(f"pred.{name}.residual_mean")
            wandb.define_metric(f"pred.{name}.residual_std")

    def __reset(self) -> None:
        self.__loss = 0.0
        self.__mse = 0.0
        self.__psnr = 0.0
        self.__ssim = 0.0
        self.__inputs = []
        self.__targets = []
        self.__outputs = []
        self.__residuals = []

    def post_train_batch(self, i: int, x: Sequence[Any], y: Sequence[Any], z: Sequence[Any]) -> None:
        wandb.log(
            {
                "train": {
                    "loss": cast(torch.Tensor, sum(self.__loss_fn(z, y))).item(),
                    "MSE": torchmetrics.functional.mean_squared_error(z[0], y[0]).item(),
                    "PSNR": torchmetrics.functional.peak_signal_noise_ratio(z[0], y[0], dim=(1, 2, 3), data_range=1.0).item(),
                    "SSIM": cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(z[0], y[0], data_range=1.0)).item(),
                }
            },
            commit=True,
        )

    def pre_val_run(self) -> None:
        self.__reset()

    def post_val_batch(self, i: int, x: Sequence[Any], y: Sequence[Any], z: Sequence[Any]) -> None:
        self.__loss += cast(torch.Tensor, sum(self.__loss_fn(z, y))).item()
        self.__mse += torchmetrics.functional.mean_squared_error(z[0], y[0]).item()
        self.__psnr += torchmetrics.functional.peak_signal_noise_ratio(z[0], y[0], dim=(1, 2, 3), data_range=1.0).item()
        self.__ssim += cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(z[0], y[0], data_range=1.0)).item()
        self.__inputs.extend([t for t in x[0][:TENSOR_LOG_COUNT]])
        self.__targets.extend([t for t in y[0][:TENSOR_LOG_COUNT]])
        self.__outputs.extend([t for t in z[0][:TENSOR_LOG_COUNT]])
        self.__residuals.extend([z[0][i : i + 1] - y[0][i : i + 1] for i in range(TENSOR_LOG_COUNT)])

    def post_val_run(self, dataset_name: str, dataloader_len: int) -> None:
        fid_metric = torchmetrics.image.fid.FrechetInceptionDistance().to(wandb.config["device"])
        fid_metric.update(torch.clamp(torch.stack(self.__outputs).repeat(1, 3, 1, 1) * 255.0, 0.0, 255.0).to(torch.uint8), real=False)
        fid_metric.update(torch.clamp(torch.stack(self.__targets).repeat(1, 3, 1, 1) * 255.0, 0.0, 255.0).to(torch.uint8), real=True)
        fid = fid_metric.compute().item()
        del fid_metric
        lat_mean = torch.stack(self.__residuals).mean().item()
        lat_std = torch.sqrt((torch.stack(self.__residuals) - lat_mean).pow(2).mean()).item()
        wandb.log(
            {
                "val": {
                    dataset_name: {
                        "loss": self.__loss / dataloader_len,
                        "MSE": self.__mse / dataloader_len,
                        "PSNR": self.__psnr / dataloader_len,
                        "SSIM": self.__ssim / dataloader_len,
                        "FID": fid,
                        "residual_mean": lat_mean,
                        "residual_std": lat_std,
                        "x": [wandb.Image(img) for img in self.__inputs[:EXAMPLE_COUNT]],
                        "y": [wandb.Image(img) for img in self.__targets[:EXAMPLE_COUNT]],
                        "z": [wandb.Image(img) for img in self.__outputs[:EXAMPLE_COUNT]],
                    }
                }
            },
            commit=True,
        )

    def pre_test_run(self) -> None:
        self.__reset()

    def post_test_batch(self, i: int, x: Sequence[Any], y: Sequence[Any], z: Sequence[Any]) -> None:
        self.__loss += cast(torch.Tensor, sum(self.__loss_fn(z, y))).item()
        self.__mse += torchmetrics.functional.mean_squared_error(z[0], y[0]).item()
        self.__psnr += torchmetrics.functional.peak_signal_noise_ratio(z[0], y[0], dim=(1, 2, 3), data_range=1.0).item()
        self.__ssim += cast(torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(z[0], y[0], data_range=1.0)).item()
        self.__inputs.extend([t for t in x[0][:TENSOR_LOG_COUNT]])
        self.__targets.extend([t for t in y[0][:TENSOR_LOG_COUNT]])
        self.__outputs.extend([t for t in z[0][:TENSOR_LOG_COUNT]])
        self.__residuals.extend([z[0][i : i + 1] - y[0][i : i + 1] for i in range(TENSOR_LOG_COUNT)])

    def post_test_run(self, dataset_name: str, dataloader_len: int) -> None:
        fid_metric = torchmetrics.image.fid.FrechetInceptionDistance().to(wandb.config["device"])
        fid_metric.update(torch.clamp(torch.stack(self.__outputs).repeat(1, 3, 1, 1) * 255.0, 0.0, 255.0).to(torch.uint8), real=False)
        fid_metric.update(torch.clamp(torch.stack(self.__targets).repeat(1, 3, 1, 1) * 255.0, 0.0, 255.0).to(torch.uint8), real=True)
        fid = fid_metric.compute().item()
        del fid_metric
        lat_mean = torch.stack(self.__residuals).mean().item()
        lat_std = torch.sqrt((torch.stack(self.__residuals) - lat_mean).pow(2).mean()).item()
        wandb.log(
            {
                "test": {
                    dataset_name: {
                        "loss": self.__loss / dataloader_len,
                        "MSE": self.__mse / dataloader_len,
                        "PSNR": self.__psnr / dataloader_len,
                        "SSIM": self.__ssim / dataloader_len,
                        "FID": fid,
                        "residual_mean": lat_mean,
                        "residual_std": lat_std,
                        "x": [wandb.Image(img) for img in self.__inputs[:EXAMPLE_COUNT]],
                        "y": [wandb.Image(img) for img in self.__targets[:EXAMPLE_COUNT]],
                        "z": [wandb.Image(img) for img in self.__outputs[:EXAMPLE_COUNT]],
                        "residual": [wandb.Image(torch.abs(self.__outputs[i] - self.__targets[i])) for i in range(EXAMPLE_COUNT)],
                    }
                },
            }
        )

    def pre_pred_run(self) -> None:
        self.__reset()

    def post_pred_batch(self, i: int, x: Sequence[Any], _: Sequence[Any], z: Sequence[Any]) -> None:
        self.__inputs.extend([t for t in x[0][:TENSOR_LOG_COUNT]])
        # self.__targets.extend([t for t in y[0][:TENSOR_LOG_COUNT]])
        self.__outputs.extend([t for t in z[0][:TENSOR_LOG_COUNT]])

    def post_pred_run(self, dataset_name: str, dataloader_len: int) -> None:
        # wandb.summary[f"pred.{dataset_name}.x"] = [wandb.Image(img) for img in self.__inputs[:EXAMPLE_COUNT]]
        # wandb.summary[f"pred.{dataset_name}.z"] = [wandb.Image(img) for img in self.__outputs[:EXAMPLE_COUNT]]
        wandb.log(
            {
                "pred": {
                    dataset_name: {
                        "x": [wandb.Image(img) for img in self.__inputs[:EXAMPLE_COUNT]],
                        "z": [wandb.Image(img) for img in self.__outputs[:EXAMPLE_COUNT]],
                    }
                },
            }
        )
