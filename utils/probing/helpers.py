import json
import os
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np

Array = np.ndarray


def load_triplets(data_root: str) -> Array:
    """Load original train and test splits for THINGS from disk and concatenate them."""
    train_triplets = np.load(os.path.join(data_root, "triplets", "train_90.npy"))
    val_triplets = np.load(os.path.join(data_root, "triplets", "test_10.npy"))
    triplets = np.concatenate((train_triplets, val_triplets), axis=0)
    return triplets.astype(int)


def partition_triplets(triplets: Array, train_objects: Array) -> Dict[str, List[int]]:
    """Partition triplets into two disjoint object sets for training and validation."""
    triplet_partitioning = defaultdict(list)
    for triplet in triplets:
        splits = list(
            map(lambda obj: "train" if obj in train_objects else "val", triplet)
        )
        if len(set(splits)) == 1:
            triplet_partitioning[splits.pop()].append(triplet.tolist())
    return triplet_partitioning


def standardize(features: Array) -> Array:
    """Center and normalize features so that they have zero-mean and unit variance."""
    return (features - features.mean(axis=0)) / features.std(axis=0)


def load_model_config(data_root: str, source: str) -> dict:
    """Load model config dictionary."""
    try:
        with open(
            os.path.join(data_root, "ts", "things", source, "model_dict.json"), "r"
        ) as f:
            model_dict = json.load(f)
    except FileNotFoundError:
        warnings.warn(f"\nMissing model config dict for models from {source}.\n")
        return None
    return model_dict


def get_temperature(
    model_config, model: str, module: str, objective: str = "cosine"
) -> List[str]:
    """Get optimal temperature values for all embeddings."""
    try:
        temperature = model_config[model][module]["temperature"][objective]
    except (KeyError, TypeError):
        temperature = 1.0
        warnings.warn(
            f"\nMissing temperature value for {model} and {module} layer.\nSetting temperature value to 1.\nThis may cause optimization problems during linear probing.\n"
        )
    return temperature
