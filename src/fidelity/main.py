import argparse
import logging
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from types import FrameType
from typing import Optional, cast

import torch
import torch.utils.data
import wandb
import wandb.sdk
import wandb.util

from fidelity.build_loss_fn import build_loss_fn
from fidelity.build_lr_scheduler import build_lr_scheduler
from fidelity.build_optimizer import build_optimizer
from fidelity.data import load_dataset
from fidelity.experiment_logger import ExperimentLogger
from fidelity.models import build_model
from fidelity.trainer import Trainer
from fidelity.utils import get_rng_states, load_config, load_rng_states, setup_determinism


torch.set_grad_enabled(False)


PROJECT_NAME = "fidelity"
CONFIG_PATH = Path("configs/config.yaml").resolve()


def main() -> None:
    #### Parse CLI arguments ####
    argparser = argparse.ArgumentParser()
    sub_argparsers = argparser.add_subparsers()
    run_argparser = sub_argparsers.add_parser("run")
    run_argparser.set_defaults(command="run")
    resume_argparser = sub_argparsers.add_parser("resume")
    resume_argparser.set_defaults(command="resume")
    resume_argparser.add_argument("run_id")
    args = argparser.parse_args()
    if not hasattr(args, "command"):
        args.command = "run"

    #### Setup logging ####
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d [%(levelname).4s][%(name)s]: %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(logging.Filter(PROJECT_NAME))
    logging.captureWarnings(True)  # TODO Fix
    warnings.filterwarnings(message=r"[^ ]+ does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True).*", action="ignore")

    #### Initialize wandb ####
    if args.command == "resume":  # TODO Test
        logging.getLogger(PROJECT_NAME).info("Initializing wandb")
        _ = wandb.init(
            project=PROJECT_NAME,
            id=args.run_id if args.command == "resume" else wandb.util.generate_id(),
            dir=tempfile.gettempdir(),
            anonymous="never",
            mode="online",
            force=True,
            resume="must",
        )
        logging.getLogger(PROJECT_NAME).info(f"Resuming run {args.run_id}")
    else:
        logging.getLogger(PROJECT_NAME).info("Initializing wandb")
        _ = wandb.init(
            project=PROJECT_NAME,
            id=wandb.util.generate_id(),
            group=None,
            job_type="training",
            tags=[],
            config=load_config(CONFIG_PATH),
            dir=tempfile.gettempdir(),
            anonymous="never",
            mode="online",
            force=True,
            resume="never",
        )
        logging.getLogger(PROJECT_NAME).info(f"Starting run {cast(wandb.sdk.wandb_run.Run, wandb.run).name}")
        logging.getLogger(PROJECT_NAME).info("Logging code to wandb")
        cast(wandb.sdk.wandb_run.Run, wandb.run).log_code(".", include_fn=lambda path: path.endswith(".py") or path.endswith(".yaml"))

    #### Setup determinism ####
    if wandb.config["seed"] is not None:
        logging.getLogger(PROJECT_NAME).info(f"Setting up determinism with seed {wandb.config['seed']}")
        setup_determinism(wandb.config["seed"])

    #### Load datasets ####
    logging.getLogger(PROJECT_NAME).info("Loading datasets")
    logging.getLogger(PROJECT_NAME).debug("Loading train dataset")
    logging.getLogger(PROJECT_NAME).debug(f"  Loading dataset {wandb.config['data']['train_dataset']['name']}")
    train_dataset = load_dataset("train")
    logging.getLogger(PROJECT_NAME).debug("Loading validation datasets")
    val_datasets = {dataset_config["name"]: logging.getLogger(PROJECT_NAME).debug(f"  Loading {dataset_config['name']}") or load_dataset("val", i) for i, dataset_config in enumerate(wandb.config["data"]["val_datasets"])}
    logging.getLogger(PROJECT_NAME).debug("Loading test datasets")
    test_datasets = {dataset_config["name"]: logging.getLogger(PROJECT_NAME).debug(f"  Loading {dataset_config['name']}") or load_dataset("test", i) for i, dataset_config in enumerate(wandb.config["data"]["test_datasets"])}
    logging.getLogger(PROJECT_NAME).debug("Loading prediction datasets")
    pred_datasets = {dataset_config["name"]: logging.getLogger(PROJECT_NAME).debug(f"  Loading {dataset_config['name']}") or load_dataset("pred", i) for i, dataset_config in enumerate(wandb.config["data"]["pred_datasets"])}

    #### Create dataloaders ####
    logging.getLogger(PROJECT_NAME).info("Creating dataloaders")
    logging.getLogger(PROJECT_NAME).debug("Creating train dataloader")
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=wandb.config["training"]["grad_batch_size"], shuffle=True, drop_last=False, generator=torch.Generator(), worker_init_fn=lambda _: setup_determinism(torch.initial_seed()))
    logging.getLogger(PROJECT_NAME).debug("Creating validation dataloaders")
    val_dataloaders = {name: torch.utils.data.DataLoader(dataset, batch_size=wandb.config["training"]["nograd_batch_size"], shuffle=False, drop_last=False, generator=torch.Generator(), worker_init_fn=lambda _: setup_determinism(torch.initial_seed())) for name, dataset in val_datasets.items()}
    logging.getLogger(PROJECT_NAME).debug("Creating test dataloaders")
    test_dataloaders = {name: torch.utils.data.DataLoader(dataset, batch_size=wandb.config["training"]["nograd_batch_size"], shuffle=False, drop_last=False, generator=torch.Generator(), worker_init_fn=lambda _: setup_determinism(torch.initial_seed())) for name, dataset in test_datasets.items()}
    logging.getLogger(PROJECT_NAME).debug("Creating prediction dataloaders")
    pred_dataloaders = {name: torch.utils.data.DataLoader(dataset, batch_size=wandb.config["training"]["nograd_batch_size"], shuffle=False, drop_last=False, generator=torch.Generator(), worker_init_fn=lambda _: setup_determinism(torch.initial_seed())) for name, dataset in pred_datasets.items()}

    #### Build model ####
    logging.getLogger(PROJECT_NAME).info("Building model")
    logging.getLogger(PROJECT_NAME).debug(f"  Constructing model {wandb.config['model']['name']}")
    model = build_model()
    model = model.to(wandb.config["device"])
    if hasattr(torch, "compile"):
        logging.getLogger(PROJECT_NAME).debug("  Compiling model")
        model = torch.compile(model)  # type: ignore
    # logging.getLogger(PROJECT_NAME).debug("  Distributing model")
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=?, output_device=?)  #TODO Launch in parallel, distribute data, distribute model, sync metrics
    logging.getLogger(PROJECT_NAME).info("Starting wandb model watchdog")
    wandb.watch(model, log_freq=100, log_graph=True)

    #### Build optimizer, learning rate scheduler, loss function ####
    logging.getLogger(PROJECT_NAME).info("Building optimizer")
    optimizer = build_optimizer(model)
    if "lr_scheduler" in wandb.config["training"]:
        logging.getLogger(PROJECT_NAME).info("Building learing rate scheduler")
    lr_scheduler = build_lr_scheduler(optimizer)
    logging.getLogger(PROJECT_NAME).info("Building loss function")
    loss_fn = build_loss_fn()

    #### Create experiment and trainer ####
    experiment_logger = ExperimentLogger(loss_fn)
    trainer = Trainer(model, optimizer, lr_scheduler, loss_fn, experiment_logger)

    #### Register signal handlers ####
    def on_slurm_timelimit(sig: int, frame: Optional[FrameType]) -> None:
        logging.getLogger(PROJECT_NAME).info("SLURM timeout")
        logging.getLogger(PROJECT_NAME).debug("  Saving state")
        artifact = wandb.Artifact("state", type="checkpoint")
        with artifact.new_file("state.ckpt", mode="wb") as file:
            torch.save(
                {
                    "RNGs": get_rng_states(),
                    "train_dataloader": cast(torch.Generator, cast(torch.utils.data.RandomSampler, train_dataloader.sampler).generator).get_state(),
                    "val_dataloader": {name: cast(torch.Generator, cast(torch.utils.data.RandomSampler, dataloader.sampler).generator).get_state() for name, dataloader in val_dataloaders.items()},
                    "test_dataloader": {name: cast(torch.Generator, cast(torch.utils.data.RandomSampler, dataloader.sampler).generator).get_state() for name, dataloader in val_dataloaders.items()},
                    "pred_dataloader": {name: cast(torch.Generator, cast(torch.utils.data.RandomSampler, dataloader.sampler).generator).get_state() for name, dataloader in val_dataloaders.items()},
                    "trainer": trainer.state_dict(),
                },
                file,
            )
        wandb.log_artifact(artifact, aliases=["latest"])
        logging.getLogger(PROJECT_NAME).debug("  Resubmitting SLRUM job")
        try:
            subprocess.run(["sbatch", "???"])
        except:
            pass
        logging.getLogger(PROJECT_NAME).info("Done")
        exit(-1)

    logging.getLogger(PROJECT_NAME).info("Registering signal handlers")
    logging.getLogger(PROJECT_NAME).debug("  Registering SLURM timeout signal handler")
    # signal.signal(signal.SIGUSR1, on_slurm_timelimit)  # TODO Implement correct handler, test

    #### Load latest state to resume / Save initial state ####
    if args.command == "resume":
        logging.getLogger(PROJECT_NAME).info("Loading previous state")
        state_artifact = wandb.use_artifact("state:latest")
        path = state_artifact.download()
        state = torch.load(os.path.join(path, "state.ckpt"))
        logging.getLogger(PROJECT_NAME).debug("  Loading RNG state")
        load_rng_states(state["RNGs"])
        logging.getLogger(PROJECT_NAME).debug("  Loading Trainer state")
        trainer.load_state_dict(state["trainer"])
        logging.getLogger(PROJECT_NAME).debug("  Loading dataloader states")
        cast(torch.Generator, cast(torch.utils.data.RandomSampler, train_dataloader.sampler).generator).set_state(state["train_dataloader"])
    else:
        logging.getLogger(PROJECT_NAME).debug("Saving initial state")
        artifact = wandb.Artifact("state", type="checkpoint")
        with artifact.new_file("state.ckpt", mode="wb") as file:
            torch.save(
                {
                    "RNGs": get_rng_states(),
                    "train_dataloader": cast(torch.Generator, cast(torch.utils.data.RandomSampler, train_dataloader.sampler).generator).get_state(),
                    "trainer": trainer.state_dict(),
                },
                file,
            )
        wandb.log_artifact(artifact, aliases=["latest", "initial", *[f"best-{name}" for name in val_dataloaders.keys()]])

    #### Train ####
    logging.getLogger(PROJECT_NAME).info("Starting training")
    trainer.train(
        train_dataloader,
        val_dataloaders,
        max_epochs=wandb.config["training"]["epochs"],
        additional_state_fn=lambda: {
            "RNGs": get_rng_states(),
            "train_dataloader": cast(torch.Generator, cast(torch.utils.data.RandomSampler, train_dataloader.sampler).generator).get_state(),
        },
    )
    logging.getLogger(PROJECT_NAME).info("Training done")

    #### Export training results ####
    logging.getLogger(PROJECT_NAME).debug("Saving latest state")
    artifact = wandb.Artifact("state", type="checkpoint")
    with artifact.new_file("state.ckpt", mode="wb") as file:
        torch.save(
            {
                "RNGs": get_rng_states(),
                "train_dataloader": cast(torch.Generator, cast(torch.utils.data.RandomSampler, train_dataloader.sampler).generator).get_state(),
                "trainer": trainer.state_dict(),
            },
            file,
        )
    wandb.log_artifact(artifact, aliases=["latest"])
    logging.getLogger(PROJECT_NAME).debug("Exporting model weights")
    artifact = wandb.Artifact("ONNX", type="onnx")
    with artifact.new_file("weights.onnx", mode="wb") as file:
        x = next(iter(val_dataloaders[next(iter(val_dataloaders.keys()))]))[0]
        x = tuple([f.to(wandb.config["device"]) if isinstance(f, torch.Tensor) else f for f in x])
        torch.onnx.export(model, x, file)
    wandb.log_artifact(artifact, aliases=["latest"])

    #### Run tests and predictions ####
    logging.getLogger(PROJECT_NAME).info("Starting testing")
    trainer.test(test_dataloaders)
    logging.getLogger(PROJECT_NAME).info("Testing done")
    logging.getLogger(PROJECT_NAME).info("Starting predicting")
    trainer.pred(pred_dataloaders)
    logging.getLogger(PROJECT_NAME).info("Predicting done")

    #### Cleanup ####
    wandb.finish()
    logging.getLogger(PROJECT_NAME).info("Done")


if __name__ == "__main__":
    os.environ["WANDB_SILENT"] = "true"
    os.environ["WANDB_CONSOLE"] = "off"
    main()
