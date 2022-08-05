import argparse
from collections import defaultdict
from email.policy import default
from gc import callbacks
import numpy as np

import os
import pickle
import torch

from torch.utils.data import DataLoader
from tqdm import tqdm
from ml_collections import config_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pytorch_lightning.callbacks import 
>>> 
from typing import Any, Dict, List, Tuple

from .data.dataset import TripletData
from .transform import LinearProbe

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any
FOLDS = ["fold_01", "fold_02", "fold_03"]


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", default="things")
    aa("--n_objects", type=int, help="Number of object categories in the data", default=1854)
    aa("--batch_size", type=int, default=256)
    aa("--epochs", type=int, 
        help="Maximum number of epochs to perform finetuning", default=100)
    aa("--burnin", type=int, 
        help="Minimum number of epochs to perform finetuning", default=10)
    aa("--patience", type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=20)
    aa("--device", type=str, default="cpu",
        choices=["cpu", "gpu"])
    aa("--results_path", type=str, help="path/to/results")
    aa("--rnd_seed", type=int, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def create_optimization_config(args) -> Tuple[FrozenDict, FrozenDict]:
    optim_cfg = config_dict.ConfigDict()
    optim_cfg.optim = args.optim
    optim_cfg.lr = args.lr
    optim_cfg.transform_dim = args.transform_dim
    optim_cfg.batch_size = args.batch_size
    optim_cfg.max_epochs = args.epochs
    optim_cfg.min_epochs = args.burnin
    optim_cfg.patience = args.patience
    optim_cfg = config_dict.FrozenConfigDict(optim_cfg)
    return optim_cfg


def load_features(results_path: str) -> Dict[str, Array]:
    with open(os.path.join(results_path, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return features


def get_batches(triplets: Tensor, batch_size: int, train: bool):
    dl = DataLoader(
        dataset=triplets,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=0,
        drop_last=False,
        pin_memory=True if train else False,
    )
    return dl

def get_callbacks(optim_cfg: FrozenDict, ckptdir: str):
    checkpoint_callback = ModelCheckpoint(dirpath=ckptdir)
    early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            mode='min',
            patience=optim_cfg.patience,
            verbose=True,
            check_finite=True,
        )
    callbacks = [checkpoint_callback, early_stopping]
    return callbacks


def load_triplet_splits(
    data_root: str, n_objects: int, split: str = "disjoint"
) -> List[Tuple[Tensor, Tensor]]:
    train_val_splits = []
    for fold in FOLDS:
        train_triplets = TripletData(
            root=data_root, n_objects=n_objects, split=split, fold=fold, train=True
        )
        val_triplets = TripletData(
            root=data_root, n_objects=n_objects, split=split, fold=fold, train=False
        )
        train_val_splits.append((train_triplets, val_triplets))
    return train_val_splits


def run(
    features: Array,
    data_root: str,
    n_objects: int,
    device: str,
    optim_cfg: FrozenDict,
    ckptdir: str,
) -> None:
    train_val_splits = load_triplet_splits(
        data_root=data_root,
        n_objects=n_objects,
    )
    callbacks = get_callbacks(
        optim_cfg=optim_cfg,
        ckptdir=ckptdir,
    )
    cv_results = {}
    for k, (train_triplets, val_triplets) in tqdm(train_val_splits, desc="Fold"):
        train_batches = get_batches(
            triplets=train_triplets,
            batch_size=optim_cfg.batch_size,
        )
        val_batches = get_batches(
            triplets=val_triplets,
            batch_size=optim_cfg.batch_size,
        )
        transformation = LinearProbe(
            features=features,
            transform_dim=optim_cfg.transform_dim,
            optim=optim_cfg.optim,
            lr=optim_cfg.lr,
        )
        trainer = Trainer(
            accelerator=device,
            callbacks=callbacks,
            max_epochs=optim_cfg.max_epochs,
            min_epochs=optim_cfg.min_epochs,
        )
        trainer.fit(transformation, train_batches, val_batches)
        val_performance = trainer.validate(
            transformation,
            dataloaders=val_batches,
        )
        cv_results[f"fold_{k:02d}"] = val_performance
    return cv_results


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    seed_everything(args.rnd_seed, workers=True)
    features = load_features(args.results_path)
    optim_cfg = create_optimization_config(args)
    cv_results = run(
        features=features,
        data_root=args.data_root,
        n_objects=args.n_objects,
        device=args.device,
        optim_cfg=optim_cfg,
    )
