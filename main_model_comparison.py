#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd

from data import DATASETS
from utils.analyses import CKA

Array = np.ndarray


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa(
        "--dataset",
        type=str,
        help="Whether to use things, things-aligned, or CIFAR-100",
        choices=DATASETS,
    )
    aa(
        "--module",
        type=str,
        help="Whether to compare odd-one-out choices of the logits or penultimate layer",
        choices=["logits", "penultimate"],
    )
    aa("--results_path", type=str, help="path/to/results")
    aa("--features_path", type=str, help="path/to/features")
    args = parser.parse_args()
    return args


def unpickle_results(results_path: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(results_path, "results.pkl"))


def get_vice_probas(data_root: str) -> Array:
    return np.load(os.path.join(data_root, "probas", "probabilities_all_triplets.npy"))


def get_vice_entropies(data_root: str) -> Array:
    return np.load(os.path.join(data_root, "entropies", "entropies_all_triplets.npy"))


def get_vice_embedding(data_root: str) -> Array:
    return np.load(os.path.join(data_root, "dimensions", "vice_embedding.npy"))


def load_features(features_path: str, data_root: str) -> Array:
    with open(os.path.join(features_path, "features.pkl"), "rb") as f:
        features = pickle.load(f)
    vice_embedding = get_vice_embedding(data_root)
    features.update(
        {"PyTorch": {"vice": {"logits": vice_embedding, "penultimate": vice_embedding}}}
    )
    return features


def add_vice(
    results: pd.DataFrame, vice_entropies: Array, vice_probas: Array
) -> pd.DataFrame:
    vice_choices = np.full_like(
        a=results[results.model == np.unique(results.model)[0]].choices.values[0],
        fill_value=2,
        dtype=int,
    )
    vice = [
        {
            "model": "vice",
            "accuracy": float(1),
            "choices": vice_choices,
            "entropies": vice_entropies,
            "probas": vice_probas,
            "source": "torch",
            "family": "VICE",
        }
    ]
    return pd.concat([results, pd.DataFrame(vice)], axis=0, ignore_index=True)


def get_agreement(choices_i, choices_j) -> float:
    assert (
        choices_i.shape[0] == choices_j.shape[0]
    ), "\nNumber of triplets needs to be equivalent to compare model choices.\n"
    triplet_agreements = np.where(choices_i == choices_j)[0]
    agreement_frac = triplet_agreements.shape[0] / choices_i.shape[0]
    return agreement_frac


def compare_model_choices(results: pd.DataFrame) -> pd.DataFrame:
    models = results.model.values
    agreements = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j:
                if np.isnan(agreements.iloc[i, j]):
                    choices_i = results[results.model == model_i].choices.values[0]
                    choices_j = results[results.model == model_j].choices.values[0]
                    agreement = get_agreement(choices_i, choices_j)
                else:
                    continue
            else:
                agreement = float(1)
            agreements.iloc[i, j] = agreement
            agreements.iloc[j, i] = agreement
    return agreements


def compare_model_representations(
    results: pd.DataFrame,
    features: Dict[str, Array],
    module: str,
    m=1854,
) -> pd.DataFrame:
    cka = CKA(m=m, kernel="linear")
    models = results.model.values
    alignments = pd.DataFrame(index=models, columns=models, dtype=float)
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                if np.isnan(alignments.iloc[i, j]):
                    X = features[results.loc[i, "source"]][models[i]][module]
                    Y = features[results.loc[j, "source"]][models[j]][module]
                    rho = cka.compare(X, Y)
                else:
                    continue
            else:
                rho = float(1)
            alignments.iloc[i, j] = rho
            alignments.iloc[j, i] = rho
    return alignments


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # unpickle results and features
    results = unpickle_results(args.results_path)
    features = load_features(args.features_path, args.data_root)
    # load vice entropies and probas
    vice_entropies = get_vice_entropies(args.data_root)
    vice_probas = get_vice_probas(args.data_root)
    # add vice to results
    results = add_vice(results, vice_entropies, vice_probas)
    # compute triplet agreements and CKA
    agreements = compare_model_choices(results)
    alignments = compare_model_representations(
        results=results, features=features, module=args.module
    )
    # save dataframes as pkl files
    agreements.to_pickle(os.path.join(args.results_path, "agreements.pkl"))
    alignments.to_pickle(os.path.join(args.results_path, "alignments.pkl"))
