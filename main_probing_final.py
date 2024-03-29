import argparse
import os
import pickle
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

import utils

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="things")
    aa("--model", type=str)
    aa(
        "--model_dict_path",
        type=str,
        default="./datasets/things/model_dict.json",
        help="Path to the model_dict.json",
    )
    aa(
        "--module",
        type=str,
        default="penultimate",
        help="neural network module for which to learn a linear transform",
        choices=["penultimate", "logits"],
    )
    aa(
        "--source",
        type=str,
        default="torchvision",
        choices=[
            "google",
            "loss",
            "custom",
            "ssl",
            "imagenet",
            "torchvision",
            "vit_same",
            "vit_best",
        ],
    )
    aa(
        "--n_objects",
        type=int,
        help="Number of object categories in the data",
        default=1854,
    )
    aa("--optim", type=str, default="Adam", choices=["Adam", "AdamW", "SGD"])
    aa("--learning_rate", type=float, default=1e-3)
    aa(
        "--lmbda",
        type=float,
        default=1e-1,
        help="Relative contribution of the regularization term",
        choices=[1e1, 1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5],
    )
    aa(
        "--batch_size",
        type=int,
        default=256,
        help="Use a power of 2 for running mini-batch SGD on GPU",
        choices=[64, 128, 256, 512, 1024],
    )
    aa(
        "--epochs",
        type=int,
        help="Maximum number of epochs to perform finetuning",
        default=100,
    )
    aa(
        "--burnin",
        type=int,
        help="Minimum number of epochs to perform finetuning",
        default=10,
        choices=[5, 10, 15, 20, 25],
    )
    aa(
        "--patience",
        type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=10,
        choices=[5, 10, 15, 20, 25, 30],
    )
    aa("--device", type=str, default="cpu", choices=["cpu", "gpu"])
    aa(
        "--num_processes",
        type=int,
        default=4,
        help="Number of devices to use for performing distributed training on CPU",
    )
    aa(
        "--use_bias",
        action="store_true",
        help="whether or not to use a bias for the naive transform",
    )
    aa("--probing_root", type=str, help="path/to/probing")
    aa("--log_dir", type=str, help="directory to checkpoint transformations")
    aa("--rnd_seed", type=int, default=42, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def create_optimization_config(args) -> Tuple[FrozenDict, FrozenDict]:
    """Create frozen config dict for optimization hyperparameters."""
    optim_cfg = dict()
    optim_cfg["optim"] = args.optim
    optim_cfg["lr"] = args.learning_rate
    optim_cfg["lmbda"] = args.lmbda
    optim_cfg["batch_size"] = args.batch_size
    optim_cfg["max_epochs"] = args.epochs
    optim_cfg["min_epochs"] = args.burnin
    optim_cfg["patience"] = args.patience
    optim_cfg["use_bias"] = args.use_bias
    optim_cfg["ckptdir"] = os.path.join(args.log_dir, args.model, args.module)
    return optim_cfg


def load_features(probing_root: str, subfolder: str = "embeddings") -> Dict[str, Array]:
    """Load features for THINGS objects from disk."""
    with open(os.path.join(probing_root, subfolder, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return features


def get_callbacks(optim_cfg: FrozenDict, steps: int = 20) -> List[Callable]:
    if not os.path.exists(optim_cfg["ckptdir"]):
        os.makedirs(optim_cfg["ckptdir"])
        print("\nCreating directory for checkpointing...\n")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=optim_cfg["ckptdir"],
        filename="ooo-finetuning-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
        every_n_epochs=steps,
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-4,
        mode="min",
        patience=optim_cfg["patience"],
        verbose=True,
        check_finite=True,
    )
    callbacks = [checkpoint_callback, early_stopping]
    return callbacks


def make_results_df(
    columns: List[str],
    probing_acc: float,
    probing_loss: float,
    model_name: str,
    module_name: str,
    source: str,
    lmbda: float,
    optim: str,
    lr: float,
    bias: bool,
) -> pd.DataFrame:
    probing_results_current_run = pd.DataFrame(index=range(1), columns=columns)
    probing_results_current_run["model"] = model_name
    probing_results_current_run["probing"] = probing_acc
    probing_results_current_run["cross-entropy"] = probing_loss
    probing_results_current_run["module"] = module_name
    probing_results_current_run["family"] = utils.analyses.get_family_name(model_name)
    probing_results_current_run["source"] = source
    probing_results_current_run["l2_reg"] = lmbda
    probing_results_current_run["optim"] = optim.lower()
    probing_results_current_run["lr"] = lr
    probing_results_current_run["bias"] = bias
    return probing_results_current_run


def save_results(args, probing_acc: float, probing_loss: float) -> None:
    out_path = os.path.join(args.probing_root, "results", "full")
    if not os.path.exists(out_path):
        print("\nCreating results directory...\n")
        os.makedirs(out_path)

    if os.path.isfile(os.path.join(out_path, "probing_results.pkl")):
        print(
            "\nFile for probing results exists.\nConcatenating current results with existing results file...\n"
        )
        probing_results_overall = pd.read_pickle(
            os.path.join(out_path, "probing_results.pkl")
        )
        probing_results_current_run = make_results_df(
            columns=probing_results_overall.columns.values,
            probing_acc=probing_acc,
            probing_loss=probing_loss,
            model_name=args.model,
            module_name=args.module,
            source=args.source,
            lmbda=args.lmbda,
            optim=args.optim,
            lr=args.learning_rate,
            bias=args.use_bias,
        )
        probing_results = pd.concat(
            [probing_results_overall, probing_results_current_run],
            axis=0,
            ignore_index=True,
        )
        probing_results.to_pickle(os.path.join(out_path, "probing_results.pkl"))
    else:
        print("\nCreating file for probing results...\n")
        columns = [
            "model",
            "probing",
            "cross-entropy",
            "module",
            "family",
            "source",
            "l2_reg",
            "optim",
            "lr",
            "bias",
        ]
        probing_results = make_results_df(
            columns=columns,
            probing_acc=probing_acc,
            probing_loss=probing_loss,
            model_name=args.model,
            module_name=args.module,
            source=args.source,
            lmbda=args.lmbda,
            optim=args.optim,
            lr=args.learning_rate,
            bias=args.use_bias,
        )
        probing_results.to_pickle(os.path.join(out_path, "probing_results.pkl"))


def run(
    features: Array,
    n_objects: int,
    data_root: str,
    device: str,
    optim_cfg: FrozenDict,
    num_processes: int,
) -> Tuple[Dict[str, List[float]], Array]:
    """Run the optimization process."""
    callbacks = get_callbacks(optim_cfg)
    # use the original train and validation splits from the THINGS data paper (Hebart et al., 2023)
    train_triplets = np.load(
        os.path.join(data_root, "triplets", "train_90.npy")
    ).tolist()
    val_triplets = np.load(os.path.join(data_root, "triplets", "test_10.npy")).tolist()
    # subtract global mean and normalize by global standard deviation
    features = (features - features.mean()) / features.std()
    # initialize transformation with small values
    optim_cfg["sigma"] = 1e-3
    train_triplets = utils.probing.TripletData(
        triplets=train_triplets,
        n_objects=n_objects,
    )
    val_triplets = utils.probing.TripletData(
        triplets=val_triplets,
        n_objects=n_objects,
    )
    train_batches = DataLoader(
        dataset=train_triplets,
        batch_size=optim_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=True,
    )
    val_batches = DataLoader(
        dataset=val_triplets,
        batch_size=optim_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    linear_probe = utils.probing.Linear(
        features=features,
        optim_cfg=optim_cfg,
    )
    trainer = Trainer(
        accelerator=device,
        callbacks=callbacks,
        # strategy="ddp_spawn" if device == "cpu" else None,
        strategy="ddp",
        max_epochs=optim_cfg["max_epochs"],
        min_epochs=optim_cfg["min_epochs"],
        devices=num_processes if device == "cpu" else "auto",
        enable_progress_bar=True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )
    trainer.fit(linear_probe, train_batches, val_batches)
    val_performance = trainer.test(
        linear_probe,
        dataloaders=val_batches,
    )
    predictions = trainer.predict(linear_probe, dataloaders=val_batches)
    # predictions = torch.cat(predictions, dim=0).tolist()
    transformation = linear_probe.transform_w.data.detach().cpu().numpy()
    if optim_cfg["use_bias"]:
        bias = linear_probe.transform_b.data.detach().cpu().numpy()
        transformation = np.concatenate((transformation, bias[:, None]), axis=1)
    return val_performance, transformation


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # seed everything for reproducibility of results
    seed_everything(args.rnd_seed, workers=True)
    features = load_features(args.probing_root)
    model_features = features[args.source][args.model][args.module]
    optim_cfg = create_optimization_config(args)
    val_performance, transform = run(
        features=model_features,
        n_objects=args.n_objects,
        data_root=args.data_root,
        device=args.device,
        optim_cfg=optim_cfg,
        num_processes=args.num_processes,
    )
    probing_acc = val_performance[0]["test_acc"]
    probing_loss = val_performance[0]["test_loss"]
    save_results(args, probing_acc=probing_acc, probing_loss=probing_loss)

    out_path = os.path.join(
        args.probing_root,
        "results",
        "full",
        args.source,
        args.model,
        args.module,
        str(args.lmbda),
        args.optim.lower(),
        str(args.learning_rate),
    )
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    with open(os.path.join(out_path, "transform.npy"), "wb") as f:
        np.save(file=f, arr=transform)
