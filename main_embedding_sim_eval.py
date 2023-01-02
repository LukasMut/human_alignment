import argparse
import os
import random
import warnings
from collections import defaultdict
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
        "--things_embeddings_path",
        type=str,
        default="/home/space/datasets/things/embeddings/model_features_per_source.pkl",
        help="path/to/things/features; necessary if you use transforms",
    )
    aa(
        "--stimulus_set",
        type=str,
        default=None,
        choices=["set1", "set2"],
        help="Similarity judgments of the dataset from King et al. (2019) were collected for two stimulus sets",
    )
    aa(
        "--category",
        type=str,
        default=None,
        choices=[
            "animals",
            "automobiles",
            "fruits",
            "furniture",
            "various",
            "vegetables",
        ],
        help="Similarity judgments of the dataset from Peterson et al. (2016) were collected for specific categories",
    )
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
        "--use_transforms",
        action="store_true",
        help="use transformation matrix obtained from linear probing on the things triplet odd-one-out task",
    )
    aa(
        "--transform_type",
        type=str,
        default="without_norm",
        choices=["without_norm", "with_norm"],
        help="type of transformation matrix being used",
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


def create_hyperparam_dicts(args) -> Tuple[FrozenDict, FrozenDict]:
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    model_cfg.source = args.embeddings_root.split("/")[-1]
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg.name = args.dataset
    data_cfg.category = args.category
    data_cfg.stimulus_set = args.stimulus_set
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def evaluate(args) -> None:
    """Evaluate the alignment of neural nets with human (pairwise) similarity judgments."""
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
        stimulus_set=args.stimulus_set if args.dataset == "free-arrangement" else None,
        object_names=object_names,
    )
    model_cfg, data_cfg = create_hyperparam_dicts(args)
    dataset = load_dataset(
        name=args.dataset,
        data_dir=data_cfg.root,
        stimulus_set=data_cfg.stimulus_set,
    )
    if args.use_transforms:
        things_features = utils.evaluation.load_features(
            path=args.things_embeddings_path
        )
        transforms = utils.evaluation.load_transforms(
            root=args.data_root, type=args.transform_type
        )
    results = []
    model_features = defaultdict(lambda: defaultdict(dict))
    for model_name, features in tqdm(embeddings.items(), desc="Model"):
        family_name = utils.analyses.get_family_name(model_name)

        if args.use_transforms:
            try:
                transform = transforms[model_cfg.source][model_name][args.module]
            except KeyError:
                warnings.warn(
                    message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                    category=UserWarning,
                )
                continue
            try:
                things_features_current_model = things_features[model_cfg.source][
                    model_name
                ][args.module]
            except KeyError:
                warnings.warn(
                    message=f"\nCould not find embedding matrix of {model_name} for the THINGS dataset.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                    category=UserWarning,
                )
                continue
            features = (
                features - things_features_current_model.mean()
            ) / things_features_current_model.std()
            features = features @ transform
            if args.transform_type == "with_norm":
                features = torch.from_numpy(features)
                features = F.normalize(features, dim=1).cpu().numpy()

        rsa_stats = utils.evaluation.perform_rsa(
            dataset=dataset,
            data_source=args.dataset,
            features=features,
        )
        spearman_rho_cosine = rsa_stats["spearman_rho_cosine_kernel"]
        spearman_rho_corr = rsa_stats["spearman_rho_corr_kernel"]
        pearson_corr_coef_cosine = rsa_stats["pearson_corr_coef_cosine_kernel"]
        pearson_corr_coef_corr = rsa_stats["pearson_corr_coef_corr_kernel"]
        if args.verbose:
            print(
                f"\nModel: {model_name}, Family: {family_name}, Spearman's rho: {spearman_rho_corr:.4f}, Pearson correlation coefficient: {pearson_corr_coef_corr:.4f}\n"
            )
        summary = {
            "model": model_name,
            "spearman_rho_cosine": spearman_rho_cosine,
            "pearson_corr_cosine": pearson_corr_coef_cosine,
            "spearman_rho_correlation": spearman_rho_corr,
            "pearson_corr_correlation": pearson_corr_coef_corr,
            "source": model_cfg.source,
            "family": family_name,
            "dataset": data_cfg.name,
            "category": data_cfg.category,
            "transform": args.use_transforms,
            "transform_type": args.transform_type if args.use_transforms else None,
        }
        results.append(summary)
        model_features[model_cfg.source][model_name][args.module] = features

    # convert results into Pandas DataFrame
    results = pd.DataFrame(results)
    out_path = os.path.join(args.out_path, args.dataset, model_cfg.source, args.module)
    if not os.path.exists(out_path):
        print("\nCreating output directory...\n")
        os.makedirs(out_path)

    # save dataframe to pickle to preserve data types after loading
    # load back with pd.read_pickle(/path/to/file/pkl)
    results.to_pickle(os.path.join(out_path, "results.pkl"))
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
