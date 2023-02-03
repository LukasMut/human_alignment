#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import warnings
from collections import defaultdict
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from ml_collections import config_dict
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from tqdm import tqdm
from torch.utils.data import Subset

import utils
from data import DATASETS, load_dataset

FrozenDict = Any
Tensor = torch.Tensor
Array = np.ndarray


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/dataset")
    aa(
        "--things_embeddings_path",
        type=str,
        default="/home/space/datasets/things/embeddings/model_features_per_source.pkl",
        help="path/to/things/features; necessary if you use transforms",
    )
    aa("--dataset", type=str, help="Which dataset to use", choices=DATASETS)
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
        "--model_names",
        type=str,
        nargs="+",
        help="models for which we want to extract featues",
    )
    aa(
        "--module",
        type=str,
        choices=["logits", "penultimate"],
        help="module for which to extract features",
    )
    aa("--overall_source", type=str, default="thingsvision")
    aa(
        "--sources",
        type=str,
        nargs="+",
        choices=[
            "custom",
            "timm",
            "torchvision",
            "vissl",
            "ssl",
        ],
        help="Source of (pretrained) models",
    )
    aa(
        "--model_dict_path",
        type=str,
        default="/home/space/datasets/things/model_dict.json",
        help="Path to the model_dict.json",
    )
    aa("--input_dim", type=int, default=224, help="input image dimensionality")
    aa(
        "--batch_size",
        metavar="B",
        type=int,
        default=118,
        help="number of images sampled during each step (i.e., mini-batch size)",
    )
    aa(
        "--out_path",
        type=str,
        default="/home/space/datasets/things/results/",
        help="path/to/results",
    )
    aa(
        "--device",
        type=str,
        default="cuda",
        help="whether evaluation should be performed on CPU or GPU (i.e., CUDA).",
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
        "--not_pretrained",
        action="store_true",
        help="load randomly initialized model instead of a pretrained model",
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
        help="whether to show print statements about model performance during training",
    )
    args = parser.parse_args()
    return args


def get_module_names(model_config, models: List[str], module: str) -> List[str]:
    """Get original module names for logits or penultimate layer."""
    module_names = []
    for model in models:
        try:
            module_name = model_config[model][module]["module_name"]
            module_names.append(module_name)
        except KeyError:
            raise Exception(
                f"\nMissing module name for {model}. Check config file and add module name.\nAborting evaluation run...\n"
            )
    return module_names


def create_config_dicts(args) -> Tuple[FrozenDict, FrozenDict]:
    """Create data and model config dictionaries."""
    model_config = utils.evaluation.load_model_config(args.model_dict_path)
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    model_cfg.names = args.model_names
    model_cfg.modules = get_module_names(model_config, model_cfg.names, args.module)
    model_cfg.sources = args.sources
    model_cfg.input_dim = args.input_dim
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg.name = args.dataset
    data_cfg.category = args.category
    data_cfg.stimulus_set = args.stimulus_set
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def evaluate(args) -> None:
    """Evaluate the alignment of neural nets with human (pairwise) similarity judgments."""
    device = torch.device(args.device)
    model_cfg, data_cfg = create_config_dicts(args)
    if args.use_transforms:
        things_features = utils.evaluation.load_features(
            path=args.things_embeddings_path
        )
        transforms = utils.evaluation.load_transforms(
            root=args.data_root, type=args.transform_type
        )
    results = []
    model_features = defaultdict(lambda: defaultdict(dict))
    for i, (model_name, source) in tqdm(
        enumerate(zip(model_cfg.names, model_cfg.sources)), desc="Model"
    ):

        if model_name.startswith("OpenCLIP"):
            name, variant, data = model_name.split("_")
            model_params = dict(variant=variant, dataset=data)
        elif model_name.startswith("clip"):
            name, variant = model_name.split("_")
            model_params = dict(variant=variant)
        else:
            name = model_name
            model_params = None

        family_name = utils.analyses.get_family_name(model_name)
        extractor = get_extractor(
            model_name=name,
            source=source,
            device=device,
            pretrained=not args.not_pretrained,
            model_parameters=model_params,
        )

        if model_name.endswith("ecoset"):
            transformations = extractor.get_transformations(
                resize_dim=128, crop_dim=128
            )
        else:
            transformations = extractor.get_transformations()
        if args.dataset == "peterson":
            transformations = Compose(
                [Lambda(lambda img: img.convert("RGB")), transformations]
            )

        dataset = load_dataset(
            name=args.dataset,
            data_dir=data_cfg.root,
            stimulus_set=data_cfg.stimulus_set,
            category=data_cfg.category,
            transform=transformations,
        )
        batches = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            backend=extractor.get_backend(),
        )
        if (
            source == "torchvision"
            and args.module == "penultimate"
            and model_name.startswith("vit")
        ):
            num_slices = len(dataset) // 2000
            subsets = [Subset(dataset, indices=indices) for indices in np.array_split(range(len(dataset)), num_slices)]
            features_list = []
            for subset in subsets:
                subset_batches = DataLoader(
                    dataset=subset,
                    batch_size=args.batch_size,
                    backend=extractor.get_backend(),
                )
                features = extractor.extract_features(
                    batches=subset_batches,
                    module_name=model_cfg.modules[i],
                    flatten_acts=False,
                )
                features = features[:, 0].copy()  # select classifier token
                features_list.append(features)
            features = np.concatenate(features_list, axis=0)
            features = features.reshape((features.shape[0], -1))
        else:
            features = extractor.extract_features(
                batches=batches,
                module_name=model_cfg.modules[i],
                flatten_acts=True,
            )

        if args.use_transforms:
            try:
                transform = transforms[source][model_name][args.module]
            except KeyError:
                warnings.warn(
                    message=f"\nCould not find transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                    category=UserWarning,
                )
                continue
            try:
                things_features_current_model = things_features[source][model_name][
                    args.module
                ]
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
        try:
            rsa_stats = utils.evaluation.perform_rsa(
                dataset=dataset,
                data_source=args.dataset,
                features=features,
            )
        except:
            warnings.warn(
                message=f"\nFound Infs or NaNs in transformation matrix for {model_name}.\nSkipping evaluation for {model_name} and continuing with next model...\n",
                category=UserWarning,
            )
            continue
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
            "source": source,
            "family": family_name,
            "dataset": data_cfg.name,
            "category": data_cfg.category,
            "transform": args.use_transforms,
            "transform_type": args.transform_type if args.use_transforms else None,
        }
        results.append(summary)
        model_features[source][model_name][args.module] = features

    # convert results into Pandas DataFrame
    results = pd.DataFrame(results)

    out_path = os.path.join(
        args.out_path, args.dataset, args.overall_source, args.module
    )
    if not os.path.exists(out_path):
        print("\nOutput directory does not exist...")
        print("Creating output directory to save results...\n")
        os.makedirs(out_path)

    # save dataframe to pickle to preserve data types after loading
    # load back with pd.read_pickle(/path/to/file/pkl)
    results.to_pickle(os.path.join(out_path, "results.pkl"))
    utils.evaluation.save_features(features=dict(model_features), out_path=out_path)


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    # set number of threads used by PyTorch if device is CPU
    if args.device.lower().startswith("cpu"):
        torch.set_num_threads(args.num_threads)
    # run evaluation script
    evaluate(args)
