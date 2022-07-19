#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unittest import result
from scipy.spatial.distance import jensenshannon
from data import DATASETS

import os
import argparse

import numpy as np
import pandas as pd

Array = np.ndarray


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)
    aa("--data_root", type=str, help="path/to/things")
    aa("--dataset", type=str, help="Which dataset to use", choices=DATASETS)
    aa("--results_path", type=str, help="path/to/results")
    args = parser.parse_args()
    return args


def unpickle_results(results_path: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(results_path, 'results.pkl'))


def get_vice_probas(data_root: str) -> Array:
    return np.load(os.path.join(data_root, 'probas', 'probabilities_correct_triplets.npy'))


def get_vice_entropies(data_root: str) -> Array:
    return np.load(os.path.join(data_root, 'entropies', 'entropies_correct_triplets.npy'))


def append_vice(results: pd.DataFrame, vice_entropies: Array, vice_probas: Array) -> pd.DataFrame:
    vice_choices = np.full_like(a=results[results.model==np.unique(results.model)[0]].choices[0], fill_value=2, dtype=int)
    vice = [{'model': 'vice', 'accuracy': float(1), 'entropies': vice_entropies, 'choices': vice_choices, 'probas': vice_probas}]
    return pd.concat([results, pd.DataFrame(vice)])


def get_agreement(choices_i, choices_j) -> float:
    assert choices_i.shape[0] == choices_j.shape[0], '\nNumber of triplets needs to be same to compare choices.\n'
    triplet_agreements = np.where(choices_i == choices_j)[0]
    agreement_frac = triplet_agreements.shape[0] / choices_i.shape[0]
    return agreement_frac

    
def compare_probabilities(probas_i: Array, probas_j: Array) -> Array:
    return np.array(list(map(lambda x: jensenshannon(x[0], x[1]), zip(probas_i, probas_j))))


def compare_models(results: pd.DataFrame, metric: str ='agreement') -> pd.DataFrame:
    models = results.model
    model_comparison = pd.DataFrame(index=models, columns=models, dtype=float)
    for i, model_i in enumerate(models):
        for j, model_j in enumerate(models):
            if i != j:
                if metric == 'agreement':
                    choices_i = results[results.model==model_i].choices.values[0]
                    choices_j = results[results.model==model_j].choices.values[0]
                    agreement = get_agreement(choices_i, choices_j)
                    model_comparison.iloc[i, j] = agreement
                else: # Jensen-Shannon distance
                    probas_i = results[results.model==model_i].probas.values[0]
                    probas_j = results[results.model==model_j].probas.values[0]
                    jsdistances = compare_probabilities(probas_i, probas_j)
                    model_comparison.iloc[i, j] = np.mean(jsdistances)
            else:
                if metric == 'agreement':
                    model_comparison.loc[model_i, model_j] = float(1)
                else:
                    model_comparison.loc[model_i, model_j] = float(0)
    return model_comparison


if __name__ == "__main__":
    # parse arguments
    args = parseargs()
    # unpickle results
    results = unpickle_results(args.results_path)
    # load vice entropies and probas
    vice_entropies = get_vice_entropies(args.data_root)
    vice_probas = get_vice_probas(args.data_root)
    # add vice to results
    results = append_vice(results, vice_entropies, vice_probas)
    # compute triplet agreements and jensen-shannon distances
    agreements = compare_models(results, metric='agreement')
    js_distances = compare_models(results, metric='jsdistance')
    # save dataframes as pkl files
    agreements.to_pickle(os.path.join(args.results_path, 'agreements.pkl'))
    js_distances.to_pickle(os.path.join(args.results_path, 'js_distances.pkl'))



