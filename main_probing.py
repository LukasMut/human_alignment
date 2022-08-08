import argparse
import os
import pickle
from typing import Any, Callable, Dict, Iterator, List, Tuple

import numpy as np
import torch
from ml_collections import config_dict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from tqdm import tqdm

import probing

Array = np.ndarray
Tensor = torch.Tensor
FrozenDict = Any


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str,
        help="Which dataset to use", default="things")
    aa("--model", type=str)
    aa("--n_objects", type=int, 
        help="Number of object categories in the data", default=1854)
    aa("--optim", type=str, default='Adam',
        choices=['Adam', 'AdamW', 'SGD'])
    aa("--learning_rate", type=float, default=1e-3)
    aa("--batch_size", type=int, default=256,
        help="Use power of 2 for running optimization on GPU",
        choices=[64, 128, 256, 512, 1024])
    aa("--transform_dim", type=int, default=100,
        help='Output dimensionality of the linear transformation',
        choices=[100, 200, 300, 400, 500])
    aa("--epochs", type=int, 
        help="Maximum number of epochs to perform finetuning", default=100)
    aa("--burnin", type=int, 
        help="Minimum number of epochs to perform finetuning", default=10)
    aa("--patience", type=int,
        help="number of checks with no improvement after which training will be stopped",
        default=10)
    aa("--device", type=str, default="cpu",
        choices=["cpu", "gpu"])
    aa("--num_processes", type=int, default=4,
        help="Number of devices to use for performing distributed training on CPU")
    aa("--results_path", type=str, help="path/to/results")
    aa("--log_dir", type=str, help='directory to checkpoint transformations')
    aa("--rnd_seed", type=int, help="random seed for reproducibility")
    args = parser.parse_args()
    return args


def create_optimization_config(args) -> Tuple[FrozenDict, FrozenDict]:
    """Create frozen config dict for optimization hyperparameters."""
    optim_cfg = config_dict.ConfigDict()
    optim_cfg.optim = args.optim
    optim_cfg.lr = args.learning_rate
    optim_cfg.transform_dim = args.transform_dim
    optim_cfg.batch_size = args.batch_size
    optim_cfg.max_epochs = args.epochs
    optim_cfg.min_epochs = args.burnin
    optim_cfg.patience = args.patience
    optim_cfg.ckptdir = os.path.join(args.log_dir, 'probing', args.model)
    optim_cfg = config_dict.FrozenConfigDict(optim_cfg)
    return optim_cfg


def load_features(results_path: str) -> Dict[str, Array]:
    """Load features for THINGS objects from disk."""
    with open(os.path.join(results_path, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    return features


def get_batches(triplets: Tensor, batch_size: int, train: bool) -> Iterator:
    dl = DataLoader(
        dataset=triplets,
        batch_size=batch_size,
        shuffle=True if train else False,
        num_workers=0,
        drop_last=False,
        pin_memory=True if train else False,
    )
    return dl

def get_callbacks(optim_cfg: FrozenDict, steps:int=20) -> List[Callable]:
    if not os.path.exists(optim_cfg.ckptdir):
        os.makedirs(optim_cfg.ckptdir)
        print('\nCreating directory for checkpointing...\n')
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=optim_cfg.ckptdir,
        filename='ooo-finetuning-epoch{epoch:02d}-val_loss{val/loss:.2f}',
        auto_insert_metric_name=False,
        every_n_epochs=steps,
    )
    early_stopping = EarlyStopping(
            monitor='val_loss',
            min_delta=1e-4,
            mode='min',
            patience=optim_cfg.patience,
            verbose=True,
            check_finite=True
    )
    callbacks = [checkpoint_callback, early_stopping]
    return callbacks

def run(
    features: Array,
    data_root: str,
    n_objects: int,
    device: str,
    optim_cfg: FrozenDict,
    rnd_seed: int,
    num_processes: int,
    k:int=3,
) -> Tuple[Dict[str, List[float]], Array]:
    """Run optimization process."""
    callbacks = get_callbacks(optim_cfg)
    triplets = probing.load_triplets(data_root)
    objects = np.arange(n_objects)
    # Perform k-fold cross-validation with k = 3 
    # NOTE: we can try k = 5, but k = 10 doesn't work
    kf = KFold(n_splits=k, random_state=rnd_seed, shuffle=True)
    cv_results = {}
    for k, (train_idx, _) in tqdm(enumerate(kf.split(objects), start=1), desc='Fold'):
        train_objects = objects[train_idx]
        # partition triplets into disjoint object sets
        triplet_partitioning = probing.partition_triplets(
            triplets=triplets, train_objects=train_objects)
        train_triplets = probing.TripletData(
            triplets=triplet_partitioning['train'], n_objects=n_objects)
        val_triplets = probing.TripletData(
            triplets=triplet_partitioning['val'], n_objects=n_objects)
        train_batches = get_batches(
            triplets=train_triplets,
            batch_size=optim_cfg.batch_size,
            train=True,
        )
        val_batches = get_batches(
            triplets=val_triplets,
            batch_size=optim_cfg.batch_size,
            train=False,
        )
        linear_probe = probing.Linear(
            features=features,
            transform_dim=optim_cfg.transform_dim,
            optim=optim_cfg.optim,
            lr=optim_cfg.lr,
        )
        trainer = Trainer(
            accelerator=device,
            callbacks=callbacks,
            strategy="ddp_spawn" if device == "cpu" else None,
            max_epochs=optim_cfg.max_epochs,
            min_epochs=optim_cfg.min_epochs,
            devices=num_processes if device == "cpu" else "auto",
            enable_progress_bar=True,
        )
        trainer.fit(linear_probe, train_batches, val_batches)
        val_performance = trainer.validate(
            linear_probe,
            dataloaders=val_batches,
        )
        cv_results[f"fold_{k:02d}"] = val_performance
    transformation = linear_probe.transform.data.detach().cpu().numpy()
    return cv_results, transformation


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # seed everything for reproducibility
    seed_everything(args.rnd_seed, workers=True)
    features = load_features(args.results_path)
    model_features = features[args.model]
    optim_cfg = create_optimization_config(args)
    cv_results, transformation = run(
        features=model_features,
        data_root=args.data_root,
        n_objects=args.n_objects,
        device=args.device,
        optim_cfg=optim_cfg,
        rnd_seed=args.rnd_seed,
        num_processes=args.num_processes,
    )
    out_path = os.path.join(args.results_path, 'probing', args.model)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    with open(os.path.join(out_path, 'transform.npy'), 'wb') as f:
        np.save(file=f, arr=transformation)

    with open(os.path.join(out_path, 'cv_results.pkl'), 'wb') as f:
        pickle.dump(obj=cv_results, file=f)
