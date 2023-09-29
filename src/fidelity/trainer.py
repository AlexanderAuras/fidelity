import os
import subprocess
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, cast

import torch
import torch.utils.data
import tqdm
import tqdm.contrib.logging
import wandb

from fidelity.build_loss_fn import LossFunc
from fidelity.experiment_logger import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],  # type: ignore
        loss_fn: LossFunc,
        experiment_logger: ExperimentLogger,
    ) -> None:
        super().__init__()
        # General attributes
        self.__model = model
        self.__optimizer = optimizer
        self.__lr_scheduler = lr_scheduler
        self.__loss_fn = loss_fn
        self.__experiment_logger = experiment_logger
        self.__epoch = 0
        self.__step = 0
        self.__duration = timedelta()
        self.__best_val_losses = None
        # State attributes for resume
        self.__train_idx = 0
        self.__val_idx = 0
        self.__test_idx = 0
        self.__pred_idx = 0
        self.__val_dataloader_name = None
        self.__test_dataloader_name = None
        self.__pred_dataloader_name = None

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader[Tuple[Sequence[Any], Sequence[Any]]],
        val_dataloaders: Mapping[str, torch.utils.data.DataLoader[Tuple[Sequence[Any], Sequence[Any]]]],
        *,
        max_steps: Optional[int] = None,
        max_epochs: Optional[int] = None,
        max_duration: Optional[timedelta] = None,
        additional_state_fn: Callable[[], Dict[str, Any]] = lambda: {},
    ) -> None:
        # General attributes
        if max_epochs is not None and (max_epochs <= 0 or self.__epoch >= max_epochs):
            return
        if max_steps is not None and (max_steps <= 0 or self.__step >= max_steps):
            return
        if max_duration is not None and (max_duration.total_seconds() <= 0 or self.__duration >= max_duration):
            return
        if self.__best_val_losses is None:
            self.__best_val_losses = self.__validate(val_dataloaders)
        continue_training = True
        while continue_training:
            # Train for ~1 epoch
            continue_training = self.__train_epoch(train_dataloader, max_steps, max_epochs, max_duration)
            # Validate
            val_losses = self.__validate(val_dataloaders)
            # Save best weights for datasets
            for name in val_dataloaders.keys():
                if val_losses[name] < self.__best_val_losses[name]:
                    self.__best_val_losses[name] = val_losses[name]
                    artifact = wandb.Artifact("state", type="checkpoint")
                    with artifact.new_file("state.ckpt", mode="wb") as file:
                        torch.save({"trainer": self.state_dict(), **additional_state_fn()}, file)
                    wandb.log_artifact(artifact, aliases=["latest", f"best-{name}"])
            # Update learning rate
            if self.__lr_scheduler is not None:
                self.__lr_scheduler.step()

    def __train_epoch(
        self,
        train_dataloader: torch.utils.data.DataLoader[Tuple[Sequence[Any], Sequence[Any]]],
        max_steps: Optional[int],
        max_epochs: Optional[int],
        max_duration: Optional[timedelta],
    ) -> bool:
        assert wandb.config["learned"]
        self.__model.train()
        # Load old state on resume
        iter_ = enumerate(train_dataloader)
        if self.__train_idx != 0:
            [next(iter_) for _ in range(self.__train_idx)]
        # Train one epoch
        with tqdm.contrib.logging.logging_redirect_tqdm():
            for i, (x, y) in tqdm.tqdm(iter_, total=len(train_dataloader), desc=f"Training {self.__epoch+1}", leave=False):
                # Load batch etc.
                self.__train_idx = i
                x = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in x])
                y = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in y])
                batch_start_time = datetime.now()
                # Forward
                with torch.enable_grad():
                    z = self.__model(*x)
                    if not isinstance(z, Tuple):
                        z = (z,)
                    loss = cast(torch.Tensor, sum(self.__loss_fn(z, y)))
                # Backward
                cast(torch.optim.Optimizer, self.__optimizer).zero_grad()
                loss.backward()
                cast(torch.optim.Optimizer, self.__optimizer).step()
                # Log
                self.__experiment_logger.post_train_batch(i, x, y, z)
                # Check stop conditions
                self.__step += 1
                self.__duration += datetime.now() - batch_start_time
                if max_steps is not None and self.__step >= max_steps:
                    return False
                if max_duration is not None and self.__duration >= max_duration:
                    return False
        self.__train_idx = 0
        self.__epoch += 1
        if max_epochs is not None and self.__epoch >= max_epochs:
            return False
        return True

    def __validate(self, val_dataloaders: Mapping[str, torch.utils.data.DataLoader[Tuple[Sequence[Any], Sequence[Any]]]]) -> Dict[str, float]:
        self.__model.eval()
        val_losses = {name: 0.0 for name in val_dataloaders.keys()}
        skip_next_dataloader = self.__val_dataloader_name is not None
        for name, dataloader in val_dataloaders.items():
            self.__experiment_logger.pre_val_run()
            # Recreate dataloader state on resume
            if skip_next_dataloader:
                if name != self.__val_dataloader_name:
                    continue
                else:
                    skip_next_dataloader = False
            self.__pred_dataloader_name = name
            iter_ = enumerate(dataloader)
            if self.__val_idx != 0:
                [next(iter_) for _ in range(self.__val_idx)]
            # Validate
            with tqdm.contrib.logging.logging_redirect_tqdm():
                for i, (x, y) in tqdm.tqdm(iter_, total=len(dataloader), desc=f"Validation {self.__epoch} ({name})", leave=False):
                    # Load batch etc.
                    self.__val_idx = i
                    x = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in x])
                    y = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in y])
                    # Forward
                    z = self.__model(*x)
                    if not isinstance(z, Tuple):
                        z = (z,)
                    # Log
                    val_losses[name] += cast(torch.Tensor, sum(self.__loss_fn(z, y))).item()
                    self.__experiment_logger.post_val_batch(i, x, y, z)
            self.__experiment_logger.post_val_run(name, len(dataloader))
            self.__val_idx = 0
        self.__val_dataloader_name = None
        return {name: val_losses[name] / len(val_dataloaders[name]) for name in val_dataloaders.keys()}

    def test(self, test_dataloaders: Mapping[str, torch.utils.data.DataLoader[Tuple[Sequence[Any], Sequence[Any]]]]) -> None:
        self.__model.eval()
        skip_next_dataloader = self.__test_dataloader_name is not None
        for name, dataloader in test_dataloaders.items():
            self.__experiment_logger.pre_test_run()
            # Recreate dataloader state on resume
            if skip_next_dataloader:
                if name != self.__test_dataloader_name:
                    continue
                else:
                    skip_next_dataloader = False
            self.__test_dataloader_name = name
            iter_ = enumerate(dataloader)
            if self.__test_idx != 0:
                [next(iter_) for _ in range(self.__test_idx)]
            # Test
            with tqdm.contrib.logging.logging_redirect_tqdm():
                for i, (x, y) in tqdm.tqdm(iter_, total=len(dataloader), desc=f"Test ({name})", leave=False):
                    # Load batch etc.
                    self.__test_idx = i
                    x = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in x])
                    y = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in y])
                    # Forward
                    z = self.__model(*x)
                    if not isinstance(z, Tuple):
                        z = (z,)
                    # Log
                    self.__experiment_logger.post_test_batch(i, x, y, z)
            self.__experiment_logger.post_test_run(name, len(dataloader))
            self.__test_idx = 0
        self.__pred_dataloader_name = None

    def pred(self, pred_dataloaders: Mapping[str, torch.utils.data.DataLoader[Tuple[Sequence[Any], Sequence[Any]]]]) -> None:
        if hasattr(self.__model, "pred"):
            self.__model.pred()  # type: ignore
        else:
            self.__model.eval()
        skip_next_dataloader = self.__pred_dataloader_name is not None
        for name, dataloader in pred_dataloaders.items():
            self.__experiment_logger.pre_pred_run()
            # Recreate dataloader state on resume
            if skip_next_dataloader:
                if name != self.__pred_dataloader_name:
                    continue
                else:
                    skip_next_dataloader = False
            self.__pred_dataloader_name = name
            iter_ = enumerate(dataloader)
            if self.__pred_idx != 0:
                [next(iter_) for _ in range(self.__pred_idx)]
            # Predict
            with tqdm.contrib.logging.logging_redirect_tqdm():
                for i, (x, y) in tqdm.tqdm(iter_, total=len(dataloader), desc=f"Predict ({name})", leave=False):
                    # Load batch etc.
                    self.__pred_idx = i
                    x = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in x])
                    y = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in y])
                    # Forward
                    z = self.__model(*x)
                    if not isinstance(z, Tuple):
                        z = (z,)
                    # Log
                    self.__experiment_logger.post_pred_batch(i, x, y, z)
            self.__experiment_logger.post_pred_run(name, len(dataloader))
            self.__pred_idx = 0
        self.__pred_dataloader_name = None

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.__model.state_dict(),
            "optimizer": self.__optimizer.state_dict() if self.__optimizer is not None else {},
            "lr_scheduler": self.__lr_scheduler.state_dict() if self.__lr_scheduler is not None else {},
            "epoch": self.__epoch,
            "step": self.__step,
            "duration": self.__duration,
            "best_val_losses": self.__best_val_losses,
            "train_idx": self.__train_idx,
            "val_idx": self.__val_idx,
            "test_idx": self.__test_idx,
            "pred_idx": self.__pred_idx,
            "val_dataloader_name": self.__val_dataloader_name,
            "test_dataloader_name": self.__test_dataloader_name,
            "pred_dataloader_name": self.__pred_dataloader_name,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__model.load_state_dict(state_dict["model"])
        if self.__optimizer is not None:
            self.__optimizer.load_state_dict(state_dict["optimizer"])
        if self.__lr_scheduler is not None:
            self.__lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        self.__epoch = state_dict["epoch"]
        self.__step = state_dict["step"]
        self.__duration = state_dict["duration"]
        self.__best_val_losses = state_dict["best_val_losses"]
        self.__val_idx = state_dict["val_idx"]
        self.__test_idx = state_dict["test_idx"]
        self.__pred_idx = state_dict["pred_idx"]
        self.__val_dataloader_name = state_dict["val_dataloader_name"]
        self.__test_dataloader_name = state_dict["test_dataloader_name"]
        self.__pred_dataloader_name = state_dict["pred_dataloader_name"]
