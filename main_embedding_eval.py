import argparse
import os
import random
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from tqdm import tqdm

import evaluation
from data import DATASETS, load_dataset

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
    aa("--module", type=str, default='penultimate',
        choices=["logits", "penultimate"],
        help="module for which to extract features")
    aa("--model_dict_path", type=str, 
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json")
    aa("--distance", type=str, default="cosine",
        choices=["cosine", "euclidean"], 
        help="distance function used for predicting the odd-one-out")
    aa("--out_path", type=str, help="path/to/results")
    aa("--num_threads", type=int, default=4,
        help="number of threads used for intraop parallelism on CPU; use only if device is CPU")
    aa("--rnd_seed", type=int, default=42,
        help="random seed for reproducibility of results")
    aa("--verbose", action="store_true",
        help="show print statements about model performance during training")
    args = parser.parse_args()
    return args


def get_temperatures(
    model_config, models: List[str], module: str, objective: str = "cosine"
) -> List[str]:
    return [model_config[model][module]["temperature"][objective] for model in models]


def create_hyperparam_dicts(args) -> Tuple[FrozenDict, FrozenDict]:
    # model_config = evaluation.load_model_config(args.model_dict_path)
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    # model_cfg.temperatures = get_temperatures(
    #    model_config, model_cfg.names, args.module
    # )
    # model_cfg = config_dict.FrozenConfigDict(model_cfg)
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
    object_names = evaluation.get_things_objects(args.data_root)
    embeddings = evaluation.load_embeddings(
        embeddings_root=args.embeddings_root, 
        object_names=object_names,
        module='embeddings' if args.module == 'penultimate' else 'logits',
        )
    model_features = dict()
    for i, (model_name, features) in tqdm(enumerate(embeddings.items()), desc="Model"):
        triplets = dataset.get_triplets()
        """
            choices, probas = evaluation.get_predictions(
                features=features, 
                triplets=triplets, 
                temperature=model_cfg.temperatures[i], 
                distance=args.distance
            )
            """
        choices, probas = evaluation.get_predictions(
            features=features,
            triplets=triplets,
            temperature=1,
            dist=args.distance,
        )
        acc = evaluation.accuracy(choices)
        entropies = evaluation.ventropy(probas)
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
    failures = evaluation.get_failures(results)

    out_path = os.path.join(args.out_path, args.embeddings_root.split('/')[-1], args.module)
    if not os.path.exists(out_path):
        print("\nCreating output directory...\n")
        os.makedirs(out_path)

    # save dataframe to pickle to preserve data types after loading
    # load back with pd.read_pickle(/path/to/file/pkl)
    results.to_pickle(os.path.join(out_path, "results.pkl"))
    failures.to_pickle(os.path.join(out_path, "failures.pkl"))
    evaluation.save_features(features=model_features, out_path=out_path)


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    torch.set_num_threads(args.num_threads)
    # run evaluation script
    evaluate(args)
