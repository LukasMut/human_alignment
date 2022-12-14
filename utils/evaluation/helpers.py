#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import itertools
import json
import os
import pickle
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from functorch import vmap

Array = np.ndarray
Tensor = torch.Tensor


def get_things_objects(data_root: str) -> Array:
    """Load name of THINGS object concepts to sort embeddings."""
    fname = "things_concepts.tsv"
    things_objects = pd.read_csv(
        os.path.join(data_root, "concepts", fname), sep="\t", encoding="utf-8"
    )
    object_names = things_objects["uniqueID"].values
    return object_names


def convert_filenames(filenames: Array) -> Array:
    """Convert binary encoded file names into strings."""
    return np.array(
        list(map(lambda f: f.decode("utf-8").split("/")[-1].split(".")[0], filenames))
    )


def load_embeddings(
    embeddings_root: str,
    module: str = "embeddings",
    sort: str = None,
    object_names: List[str] = None,
) -> Dict[str, Array]:
    """Load Google internal embeddings and sort them according to THINGS object sorting."""

    def get_order(sorted_names: List[str]) -> Array:
        """Get correct order of file names."""
        order = np.array([np.where(filenames == n)[0][0] for n in sorted_names])
        return order

    embeddings = {}
    for f in os.scandir(embeddings_root):
        fname = f.name
        model = fname.split(".")[0]
        with open(os.path.join(embeddings_root, fname), "rb") as f:
            embedding_file = pickle.load(f)
            embedding = embedding_file[module]
            if sort:
                filenames = embedding_file["filenames"]
                filenames = convert_filenames(filenames)
                if sort == "things":
                    assert isinstance(
                        object_names, Union[list, Array]
                    ), "\nTo sort features according to things object names, a list (or an array) of object names is required.\n"
                    order = get_order(object_names)
                else:
                    order = get_order(sorted(filenames))
                embedding_sorted = embedding[order]
                embeddings[model] = embedding_sorted
            else:
                embeddings[model] = embedding
    return embeddings


def compute_dots(triplet: Tensor, pairs: List[Tuple[int]]) -> Tensor:
    dots = torch.tensor([triplet[i] @ triplet[j] for i, j in pairs])
    return dots


def compute_distances(triplet: Tensor, pairs: List[Tuple[int]], dist: str) -> Tensor:
    if dist == "cosine":
        dist_fun = lambda u, v: 1 - F.cosine_similarity(u, v, dim=0)
    elif dist == "euclidean":
        dist_fun = lambda u, v: torch.linalg.norm(u - v, ord=2)
    elif dist == "dot":
        dist_fun = lambda u, v: -torch.dot(u, v)
    else:
        raise Exception(
            "\nDistance function other than Cosine or Euclidean distance is not yet implemented\n"
        )
    distances = torch.tensor([dist_fun(triplet[i], triplet[j]) for i, j in pairs])
    return distances


def get_predictions(
    features: Array, triplets: Array, temperature: float = 1.0, dist: str = "cosine"
) -> Tuple[Tensor, Tensor]:
    """Get the odd-one-out choices for a given model."""
    features = torch.from_numpy(features)
    indices = {0, 1, 2}
    pairs = list(itertools.combinations(indices, r=2))
    choices = torch.zeros(triplets.shape[0])
    probas = torch.zeros(triplets.shape[0], len(indices))
    print(f"\nShape of embeddings {features.shape}\n")
    for s, (i, j, k) in enumerate(triplets):
        triplet = torch.stack([features[i], features[j], features[k]])
        distances = compute_distances(triplet, pairs, dist)
        dots = compute_dots(triplet, pairs)
        most_sim_pair = pairs[torch.argmin(distances).item()]
        ooo_idx = indices.difference(most_sim_pair).pop()
        choices[s] += ooo_idx
        probas[s] += F.softmax(dots * temperature, dim=0)
    return choices, probas


def accuracy(choices: List[bool], target: int = 2) -> float:
    """Computes the odd-one-out triplet accuracy."""
    return round(torch.where(choices == target)[0].shape[0] / choices.shape[0], 4)


def ventropy(probabilities: Tensor) -> Tensor:
    """Computes the entropy for a batch of (discrete) probability distributions."""

    def entropy(p: Tensor) -> Tensor:
        return -(
            torch.where(p > torch.tensor(0.0), p * torch.log(p), torch.tensor(0.0))
        ).sum()

    return vmap(entropy)(probabilities)


def get_model_choices(results: pd.DataFrame) -> Array:
    models = results.model.unique()
    model_choices = np.stack(
        [results[results.model == model].choices.values[0] for model in models],
        axis=1,
    )
    return model_choices


def filter_failures(model_choices: Array, target: int = 2):
    """Filter for triplets where every model predicted differently than humans."""
    failures, choices = zip(
        *list(filter(lambda kv: target not in kv[1], enumerate(model_choices)))
    )
    return failures, np.asarray(choices)


def get_failures(results: pd.DataFrame) -> pd.DataFrame:
    model_choices = get_model_choices(results)
    failures, choices = filter_failures(model_choices)
    model_failures = pd.DataFrame(
        data=choices, index=failures, columns=results.model.unique()
    )
    return model_failures


def save_features(features: Dict[str, Array], out_path: str) -> None:
    """Pickle dictionary of model features and save it to disk."""
    with open(os.path.join(out_path, "features.pkl"), "wb") as f:
        pickle.dump(features, f)


def load_model_config(path: str) -> dict:
    """Load model config file."""
    with open(path, "r") as f:
        model_dict = json.load(f)
    return model_dict


def load_transforms(root: str, type: str, format: str = "pkl") -> Dict[str, Dict[str, Dict[str, Array]]]:
    """Load transformation matrices obtained from linear probing on things triplet odd-one-out task into memory."""
    transforms_subdir = os.path.join(root, "transforms")
    for f in os.scandir(transforms_subdir):
        if f.is_file():
            f_name = f.name
            if f_name.endswith(format):
                if type in f_name:
                    with open(os.path.join(transforms_subdir, f_name), "rb") as f:
                        transforms = pickle.load(f)
                        break
    return transforms
