#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from functorch import vmap

Array = np.ndarray
Tensor = torch.Tensor


def get_things_objects(data_root: str) -> Array:
    fname = "things_concepts.tsv"
    things_objects = pd.read_csv(
        os.path.join(data_root, "concepts", fname), sep="\t", encoding="utf-8"
    )
    object_names = things_objects["uniqueID"].values
    return object_names


def convert_filenames(filenames: Array) -> Array:
    return np.array(
        list(map(lambda f: f.decode("utf-8").split("/")[-1].split(".")[0], filenames))
    )


def load_embeddings(embeddings_root: str, object_names: str) -> Dict[str, Array]:
    embeddings = {}
    for f in os.scandir(embeddings_root):
        fname = f.name
        model = fname.split(".")[0]
        with open(os.path.join(embeddings_root, fname), "rb") as f:
            embedding_file = pickle.load(f)
            embedding = embedding_file["embeddings"]
            filenames = embedding_file["filenames"]
            filenames = convert_filenames(filenames)
            things_sorting = np.array(
                [np.where(filenames == n)[0][0] for n in object_names]
            )
            embedding_sorted = embedding[things_sorting]
            embeddings[model] = embedding_sorted
    return embeddings


def compute_dots(triplet: Tensor, pairs: List[Tuple[int]]) -> Tensor:
    dots = torch.tensor([triplet[i] @ triplet[j] for i, j in pairs])
    return dots


def compute_distances(triplet: Tensor, pairs: List[Tuple[int]], dist: str) -> Tensor:
    if dist == "cosine":
        dist_fun = lambda u, v: 1 - F.cosine_similarity(u, v, dim=0)
    elif dist == "euclidean":
        dist_fun = lambda u, v: torch.linalg.norm(u - v, ord=2)
    else:
        raise Exception(
            "\nDistance function other than Cosine or Euclidean distance is not yet implemented\n"
        )
    distances = torch.tensor([dist_fun(triplet[i], triplet[j]) for i, j in pairs])
    return distances


def get_predictions(
    features: Array, triplets: Array, temperature: float = 1.0, dist: str = "cosine"
) -> Tuple[Tensor, Tensor]:
    features = torch.from_numpy(features)
    pairs = [(0, 1), (0, 2), (1, 2)]
    indices = {0, 1, 2}
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


def get_model_choices(results: pd.DataFrame):
    models = results.model.unique()
    model_choices = np.stack(
        [results[results.model == model].choices.values[0] for model in models],
        axis=1,
    )
    return model_choices


def filter_failures(model_choices: Array, target: int = 2):
    """Filter for triplets where every model predicted differently from humans."""
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
