

import pickle
import os
from typing import Any, Dict, List
import numpy as np
import pandas as pd

Array = np.ndarray



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any,  Dict, List, Tuple
from tqdm import tqdm
from ml_collections import config_dict
from data import load_dataset, DATASETS

import pickle
import os
import random
import re
import torch
import argparse
import utils
import json

import numpy as np
import pandas as pd
import torch.nn.functional as F

FrozenDict = Any
Tensor = torch.Tensor
Array = np.ndarray


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--embeddings_root", type=str, help="path/to/embeddings")
    aa("--dataset", type=str, help="Which dataset to use", choices=DATASETS)
    aa("--module", type=str,
        choices=["logits", "penultimate"],
        help="module for which to extract features")
    aa("--model_dict_path", type=str, 
        default="/home/space/datasets/things/model_dict.json", 
        help="Path to the model_dict.json")
    aa("--distance", type=str, default="cosine", 
        choices=["cosine", "euclidean"], 
        help="distance function used for predicting the odd-one-out")
    aa("--input_dim", type=int, default=224, help="input image dimensionality")
    aa("--batch_size", metavar="B", type=int, default=128,
        help="number of triplets sampled during each step (i.e., mini-batch size)")
    aa("--out_path", type=str, help="path/to/results")
    aa("--num_threads", type=int, default=4,
        help="number of threads used for intraop parallelism on CPU; use only if device is CPU")
    aa("--rnd_seed", type=int, default=42,
        help="random seed for reproducibility of results")
    aa("--verbose", action="store_true",
        help="whether to show print statements about model performance during training")
    args = parser.parse_args()
    return args


def load_model_config(path: str) -> dict:
    with open(path, "r") as f:
        model_dict = json.load(f)
    return model_dict


def get_temperatures(
    model_config, models: List[str], module: str, objective: str = "cosine"
) -> List[str]:
    return [model_config[model][module]["temperature"][objective] for model in models]


def create_hyperparam_dicts(args) -> Tuple[FrozenDict, FrozenDict]:
    model_config = load_model_config(args.model_dict_path)
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    # model_cfg.temperatures = get_temperatures(
    #    model_config, model_cfg.names, args.module
    # )
    model_cfg.input_dim = args.input_dim
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def evaluate(args) -> None:
    model_cfg, data_cfg = create_hyperparam_dicts(args)
    dataset = load_dataset(
            name=args.dataset,
            data_dir=data_cfg.root,
        )
    results = []
    object_names = utils.get_things_objects(args.data_root)
    embeddings = utils.load_embeddings(args.embeddings_root, object_names)
    model_features = dict()
    for i, (model_name, features) in tqdm(enumerate(embeddings.items()), desc='Model'):
            triplets = dataset.get_triplets()
            choices, probas = utils.get_predictions(
                features, triplets, model_cfg.temperatures[i], args.distance
            )
            acc = utils.accuracy(choices)
            entropies = utils.ventropy(probas)
            mean_entropy = entropies.mean().item()
            if args.verbose:
                print(
                    f"\nModel: {model_name}, Accuracy: {acc:.4f}, Average triplet entropy: {mean_entropy:.3f}\n"
                )
            summary = {
                "model": model_name,
                "accuracy": acc,
                "choices": choices.cpu().numpy(),
                "entropies": entropies.cpu().numpy(),
                "probas": probas.cpu().numpy(),
            }
            results.append(summary)
            model_features[model_name] = features

    # convert results into Pandas DataFrame
    results = pd.DataFrame(results)
    failures = utils.get_failures(results)

    if not os.path.exists(args.out_path):
        print("\nCreating output directory...\n")
        os.makedirs(args.out_path)

    # save dataframe to pickle to preserve data types after loading
    # load back with pd.read_pickle(/path/to/file/pkl)
    results.to_pickle(os.path.join(args.out_path, "results.pkl"))
    failures.to_pickle(os.path.join(args.out_path, "failures.pkl"))
    utils.save_features(features=model_features, out_path=args.out_path)


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    torch.set_num_threads(args.num_threads)
    # run evaluation script
    evaluate(args)








