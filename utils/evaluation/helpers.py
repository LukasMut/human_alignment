#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy
import itertools
import json
import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F
from functorch import vmap
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from thingsvision.core.rsa.helpers import correlation_matrix, cosine_matrix

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
    stimulus_set: str = None,
    object_names: List[str] = None,
) -> Dict[str, Array]:
    """Load Google internal embeddings and sort them according to THINGS object sorting."""

    def get_order(filenames: List[str], sorted_names: List[str]) -> Array:
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
                    try:
                        assert isinstance(
                            object_names, Union[List[str], Array]
                        ), "\nTo sort features according to the THINGS object names, a list (or an array) of object names is required.\n"
                    except TypeError:
                        print("\nChecking type of object names failed.\n")
                    order = get_order(object_names)
                else:  # alphanumeric sorting for multi-arrangement data
                    if stimulus_set:
                        sorted_names = sorted(
                            list(
                                filter(lambda x: x.startswith(stimulus_set), filenames)
                            )
                        )
                    else:
                        sorted_names = sorted(copy.deepcopy(filenames))
                    order = get_order(filenames, sorted_names)
                embedding_sorted = embedding[order]
                embeddings[model] = embedding_sorted
            else:
                embeddings[model] = embedding
    return embeddings


def compute_dots(triplet: Tensor, pairs: List[Tuple[int]]) -> Tensor:
    return torch.tensor([triplet[i] @ triplet[j] for i, j in pairs])


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
        if torch.unique(distances).shape[0] == 1:
            # If all distances are the same, we set the index to -1 (i.e., signifies an incorrect choice)
            choices[s] += -1
        else:
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


def load_features(path: str) -> Dict[str, Dict[str, Dict[str, Array]]]:
    assert os.path.isfile(path) and path.endswith(
        ".pkl"
    ), "\nThe provided path to features for THINGS is not a valid path.\nPlease provide a valid path.\n"
    with open(path, "rb") as f:
        features = pickle.load(f)
    return features


def load_transforms(
    root: str, type: str, format: str = "pkl"
) -> Dict[str, Dict[str, Dict[str, Array]]]:
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


def perform_rsa(dataset: Any, data_source: str, features: Array) -> Dict[str, float]:
    if data_source == "free-arrangement":
        cosine_rdm_dnn = compute_rdm(features, method="cosine")
        corr_rdm_dnn = compute_rdm(features, method="correlation")
        triu_inds = np.triu_indices(corr_rdm_dnn.shape[0], k=1)
        pairwise_dists_cosine = cosine_rdm_dnn[triu_inds]
        pairwise_dists_corr = corr_rdm_dnn[triu_inds]
        pairwise_dists_human = dataset.pairwise_dists
        spearman_rho_cosine = scipy.stats.spearmanr(
            pairwise_dists_cosine, pairwise_dists_human
        )[0]
        pearson_corr_coef_cosine = scipy.stats.pearsonr(
            pairwise_dists_cosine, pairwise_dists_human
        )[0]
        spearman_rho_corr = scipy.stats.spearmanr(
            pairwise_dists_corr, pairwise_dists_human
        )[0]
        pearson_corr_coef_corr = scipy.stats.pearsonr(
            pairwise_dists_corr, pairwise_dists_human
        )[0]
    else:
        if data_source == "peterson":
            cosine_rdm_dnn = cosine_matrix(features)
            corr_rdm_dnn = correlation_matrix(features)
            rdm_humans = dataset.get_rsm()
        else:
            cosine_rdm_dnn = compute_rdm(features, method="cosine")
            corr_rdm_dnn = compute_rdm(features, method="correlation")
            rdm_humans = dataset.get_rdm()
        spearman_rho_cosine = correlate_rdms(
            cosine_rdm_dnn, rdm_humans, correlation="spearman"
        )
        pearson_corr_coef_cosine = correlate_rdms(
            cosine_rdm_dnn, rdm_humans, correlation="pearson"
        )
        spearman_rho_corr = correlate_rdms(
            corr_rdm_dnn, rdm_humans, correlation="spearman"
        )
        pearson_corr_coef_corr = correlate_rdms(
            corr_rdm_dnn, rdm_humans, correlation="pearson"
        )
    rsa_stats = {}
    rsa_stats["spearman_rho_cosine_kernel"] = spearman_rho_cosine
    rsa_stats["spearman_rho_corr_kernel"] = spearman_rho_corr
    rsa_stats["pearson_corr_coef_cosine_kernel"] = pearson_corr_coef_cosine
    rsa_stats["pearson_corr_coef_corr_kernel"] = pearson_corr_coef_corr
    return rsa_stats
