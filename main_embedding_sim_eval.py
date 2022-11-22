import argparse
import os
import random
import warnings
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
from ml_collections import config_dict
from thingsvision.core.extraction import center_features
from thingsvision.core.rsa import compute_rdm, correlate_rdms
from thingsvision.core.rsa.helpers import correlation_matrix, cosine_matrix
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
    aa("--category", type=str, default=None,
        choices=["animals", "automobiles", "fruits", "furniture", "various", "vegetables"],
        help="Similarity judgments of the dataset from Peterson et al. (2016) were collected for specific categories",
    )
    aa("--module", type=str, default="penultimate",
        choices=["logits", "penultimate"],
        help="module for which to extract features")
    aa("--model_dict_path", type=str,
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json")
    aa("--distance", type=str, default="cosine",
        choices=["cosine", "euclidean"],
        help="distance function used for predicting the odd-one-out")
    aa("--out_path", type=str, help="path/to/results",
        default="/home/space/datasets/things/results")
    aa("--num_threads", type=int, default=4,
        help="number of threads used for intraop parallelism on CPU; use only if device is CPU")
    aa("--use_transforms", action="store_true",
        help="use transformation matrix obtained from linear probing on the things triplet odd-one-out task")
    aa("--rnd_seed", type=int, default=42,
        help="random seed for reproducibility of results")
    aa("--verbose", action="store_true",
        help="show print statements about model performance during training")
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
    data_cfg.name = args.dataset
    data_cfg.category = args.category
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def evaluate(args) -> None:
    """Perform evaluation of embeddings with optimal temperature values."""
    if args.cifar100:
        sort = None
        object_names = None
    elif args.dataset == "things":
        sort = "things"
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
    if args.use_transforms:
        transforms = utils.evaluation.load_transforms(args.data_root)

    results = []
    model_features = dict()
    for model_name, features in tqdm(embeddings.items(), desc="Model"):
        family_name = utils.analyses.get_family_name(model_name)

        if args.use_transforms:
            try:
                transform = transforms[model_cfg.source][model_name][args.module]
            except KeyError:
                warnings.warn(
                    message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name}\n",
                    category=UserWarning,
                    )
                continue
            features = utils.probing.standardize(features)
            features = features @ transform
        else:
            # NOTE: should we center or standardize (i.e., z-transform) feature matrix for zero-shot eval?
            # features = utils.probing.standardize(features)
            features = center_features(features)

        if args.dataset == "peterson":
            cosine_rdm_dnn = cosine_matrix(features)
            corr_rdm_dnn = correlation_matrix(features)
            rdm_humans = dataset.get_rsm()
        else:
            cosine_rdm_dnn = compute_rdm(features, method="cosine")
            corr_rdm_dnn = compute_rdm(features, method="correlation")
            rdm_humans = dataset.get_rdm()

        spearman_rho_cosine = correlate_rdms(cosine_rdm_dnn, rdm_humans, correlation="spearman")
        pearson_corr_coef_cosine = correlate_rdms(cosine_rdm_dnn, rdm_humans, correlation="pearson")

        spearman_rho_corr = correlate_rdms(corr_rdm_dnn, rdm_humans, correlation="spearman")
        pearson_corr_coef_corr = correlate_rdms(corr_rdm_dnn, rdm_humans, correlation="pearson")

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
    utils.evaluation.save_features(features=model_features, out_path=out_path)


if __name__ == "__main__":
    # parse arguments and set all random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    torch.set_num_threads(args.num_threads)
    # run evaluation script
    evaluate(args)
