import argparse
import os
import random
import warnings
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from tqdm import tqdm

import utils
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
    aa(
        "--module",
        type=str,
        default="penultimate",
        choices=["logits", "penultimate"],
        help="module for which to extract features",
    )
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json",
    )
    aa(
        "--distance",
        type=str,
        default="cosine",
        choices=["cosine", "euclidean"],
        help="distance function used for predicting the odd-one-out",
    )
    aa(
        "--out_path",
        type=str,
        help="path/to/results",
        default="/home/space/datasets/things/results",
    )
    aa(
        "--num_threads",
        type=int,
        default=4,
        help="number of threads used for intraop parallelism on CPU; use only if device is CPU",
    )
    aa(
        "--rnd_seed",
        type=int,
        default=42,
        help="random seed for reproducibility of results",
    )
    aa(
        "--verbose",
        action="store_true",
        help="show print statements about model performance during training",
    )
    aa("--cifar100", action="store_true")
    args = parser.parse_args()
    return args


def get_temperatures(
    model_config, models: List[str], module: str, objective: str = "cosine"
) -> List[str]:
    """Get optimal temperature values for all embeddings."""
    temperatures = []
    for model in models:
        try:
            t = model_config[model][module]["temperature"][objective]
        except KeyError:
            t = 1.0
            warnings.warn(
                f"\nMissing temperature value for {model} and {module} layer.\nSetting temperature value to 1.\n"
            )
        temperatures.append(t)
    return temperatures


def create_hyperparam_dicts(args, model_names) -> Tuple[FrozenDict, FrozenDict]:
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()

    try:
        model_config = utils.evaluation.load_model_config(args.model_dict_path)
        model_cfg.temperatures = get_temperatures(
            model_config, model_names, args.module
        )
    except FileNotFoundError:
        model_cfg.temperatures = np.ones(len(model_names), dtype=np.float64)
        warnings.warn(
            f"\nCould not find model config dict in {args.model_dict_path}.\nSetting temperature values to 1.0.\n"
        )
    model_cfg.source = args.embeddings_root.split("/")[-1]
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def evaluate(args) -> None:
    """Perform evaluation of embeddings with optimal temperature values."""
    if args.cifar100:
        sort = None
        object_names = None
    elif args.dataset == "things":
        sort = args.dataset
        object_names = utils.evaluation.get_things_objects(args.data_root)
    else:
        sort = "alphanumeric"
        object_names = None

    embeddings = utils.evaluation.load_embeddings(
        embeddings_root=args.embeddings_root,
        module="embeddings" if args.module == "penultimate" else "logits",
        sort=sort,
        object_names=object_names,
    )
    model_cfg, data_cfg = create_hyperparam_dicts(args, embeddings.keys())
    dataset = load_dataset(
        name=args.dataset,
        data_dir=data_cfg.root,
    )
    results = []

    model_features = defaultdict(lambda: defaultdict(dict))
    for i, (model_name, features) in tqdm(enumerate(embeddings.items()), desc="Model"):
        family = utils.analyses.get_family_name(model_name)
        triplets = dataset.get_triplets()
        choices, probas = utils.evaluation.get_predictions(
            features=features,
            triplets=triplets,
            temperature=model_cfg.temperatures[i],
            dist=args.distance,
        )

        acc = utils.evaluation.accuracy(choices)
        entropies = utils.evaluation.ventropy(probas)
        mean_entropy = entropies.mean().item()
        if args.verbose:
            print(
                f"\nModel: {model_name}, Family: {family}, Zero-shot accuracy: {acc:.4f}, Average triplet entropy: {mean_entropy:.3f}\n"
            )
        summary = {
            "model": model_name,
            "zero-shot": acc,
            "choices": choices.cpu().numpy(),
            "entropies": entropies.cpu().numpy(),
            "probas": probas.cpu().numpy(),
            "source": model_cfg.source,
            "family": family,
        }
        results.append(summary)
        model_features[model_cfg.source][model_name][args.module] = features

    # convert results into Pandas DataFrame
    results = pd.DataFrame(results)
    failures = utils.evaluation.get_failures(results)

    out_path = os.path.join(args.out_path, args.dataset, model_cfg.source, args.module)
    if not os.path.exists(out_path):
        print("\nCreating output directory...\n")
        os.makedirs(out_path)

    # save dataframe to pickle to preserve data types after loading
    # load back with pd.read_pickle(/path/to/file/pkl)
    results.to_pickle(os.path.join(out_path, "results.pkl"))
    failures.to_pickle(os.path.join(out_path, "failures.pkl"))
    utils.evaluation.save_features(features=dict(model_features), out_path=out_path)


if __name__ == "__main__":
    # parse arguments and set all random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    torch.set_num_threads(args.num_threads)
    # run evaluation script
    evaluate(args)
