#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader
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
    aa("--dataset", type=str, help="Which dataset to use", choices=DATASETS)
    aa("--model", type=str, help="model for which we want to extract features")
    aa("--layers", type=str, nargs="+", help="module for which to extract features")
    aa(
        "--source",
        type=str,
        default="torchvision",
        choices=["timm", "torchvision", "custom", "vissl"],
        help="Source of (pretrained) models",
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
        help="distance function used to predict the odd-one-out",
    )
    aa("--input_dim", type=int, default=224, help="input image dimensionality")
    aa(
        "--batch_size",
        metavar="B",
        type=int,
        default=128,
        help="number of triplets sampled during each step (i.e., mini-batch size)",
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
    aa(
        "--not_pretrained",
        action="store_true",
        help="load random model instead of pretrained",
    )
    args = parser.parse_args()
    return args


def evaluate(args) -> None:
    device = torch.device(args.device)
    results = []
    model_features = {}
    family_name = utils.analyses.get_family_name(args.model)
    extractor = get_extractor(
        model_name=args.model,
        source=args.source,
        device=device,
        pretrained=not args.not_pretrained,
    )
    dataset = load_dataset(
        name=args.dataset,
        data_dir=args.data_root,
        transform=extractor.get_transformations(),
    )
    batches = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        backend=extractor.get_backend(),
    )
    for module_name in tqdm(args.layers, desc="Layer"):
        features = extractor.extract_features(
            batches=batches,
            module_name=module_name,
            flatten_acts=True,
        )
        if len(features.shape) >= 3:
            # global average pooling
            features = features.mean(axis=-1).mean(axis=-1)
        triplets = dataset.get_triplets()
        choices, probas = utils.evaluation.get_predictions(
            features=features,
            triplets=triplets,
            temperature=float(1),
            dist=args.distance,
        )
        acc = utils.evaluation.accuracy(choices)
        entropies = utils.evaluation.ventropy(probas)
        mean_entropy = entropies.mean().item()
        if args.verbose:
            print(
                f"\nModel: {args.model}, Layer {module_name}, Accuracy: {acc:.4f}, Average triplet entropy: {mean_entropy:.3f}\n"
            )
        summary = {
            "model": args.model,
            "layer": module_name,
            "zero-shot": acc,
            "choices": choices.cpu().numpy(),
            "entropies": entropies.cpu().numpy(),
            "probas": probas.cpu().numpy(),
            "source": args.source,
            "family": family_name,
            "dataset": args.dataset,
        }
        results.append(summary)
        model_features[module_name] = features

    # convert results into Pandas DataFrame
    results = pd.DataFrame(results)

    out_path = args.out_path
    if not os.path.exists(out_path):
        print("\nCreating output directory...\n")
        os.makedirs(out_path)

    # save dataframe to pickle to preserve data types after loading
    # load back with pd.read_pickle(/path/to/file/pkl)
    results.to_pickle(os.path.join(out_path, "results.pkl"))
    utils.evaluation.save_features(features=model_features, out_path=out_path)


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
