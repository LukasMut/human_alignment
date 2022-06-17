#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, List, Tuple
from tqdm import tqdm
from ml_collections import config_dict
from thingsvision.model_class import Model
from torch.utils.data import DataLoader
from functorch import vmap
from data import THINGSBehavior

import os
import random
import re
import torch
import argparse

import numpy as np
import pandas as pd
import torch.nn.functional as F
import thingsvision.vision as vision


FrozenDict = Any
Tensor = torch.Tensor
Array = np.ndarray


def parseargs():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data_root", type=str, help="path/to/things")
    aa("--model_names", type=str, nargs="+",
        help="models for which we want to extract featues")
    aa("--module_names", type=str, nargs="+",
        help="modules of models for which to extract features")
    aa("--input_dim", type=int, default=224, help="input image dimensionality")
    aa("--batch_size", metavar="B", type=int, default=128,
        help="number of triplets sampled during each step (i.e., mini-batch size)")
    aa("--out_path", type=str, help="path/to/results")
    aa("--device", type=str, default="cuda",
        help="whether evaluation should be performed on CPU or GPU (i.e., CUDA).")
    aa("--num_threads", type=int, default=4,
        help="number of threads used for intraop parallelism on CPU; use only if device is CPU")
    aa("--temperature", type=float, default=1.,
        choices=[1., 0.1, 0.01, 0.001, 0.0001],
        help='temperature scaling (i.e., beta param in softmax function)')
    aa("--rnd_seed", type=int, default=42,
        help="random seed for reproducibility of results")
    aa("--verbose", action="store_true",
        help="whether to display print statements about model performance during training")
    args = parser.parse_args()
    return args


def create_hyperparam_dicts(args) -> Tuple[FrozenDict, FrozenDict]:
    model_cfg = config_dict.ConfigDict()
    data_cfg = config_dict.ConfigDict()
    model_cfg.names = args.model_names
    model_cfg.modules = args.module_names
    model_cfg.input_dim = args.input_dim
    model_cfg = config_dict.FrozenConfigDict(model_cfg)
    data_cfg.root = args.data_root
    data_cfg = config_dict.FrozenConfigDict(data_cfg)
    return model_cfg, data_cfg


def compute_similarities(triplet: Tensor, pairs: List[Tuple[int]], sim: str = 'dot') -> Tensor:
    if sim == 'dot':
        sim_fun = lambda u, v: u @ v
    elif sim == 'cosine':
        sim_fun = lambda u, v: F.cosine_similarity(u, v, dim=0)
    else:
        raise Exception('Similarity function other than dot product or cosine sim is not yet implemented')
    similarities = torch.tensor(
        [sim_fun(triplet[i], triplet[j]) for i, j in pairs]
    )
    return similarities


def get_predictions(features: Array, triplets: Array, temperature: float) -> List[bool]:
    features = torch.from_numpy(features)
    pairs = [(0, 1), (0, 2), (1, 2)]
    indices = {0, 1, 2}
    choices = torch.zeros(triplets.shape[0])
    probas = torch.zeros(triplets.shape[0], len(indices))
    print(f"\nShape of embeddings {features.shape}\n")
    for s, (i, j, k) in enumerate(triplets):
        triplet = torch.stack([features[i], features[j], features[k]])
        cosine_similarities = compute_similarities(triplet, pairs, sim='cosine')
        dots = compute_similarities(triplet, pairs, sim='dot')
        most_sim_pair = pairs[torch.argmax(cosine_similarities).item()]
        ooo_idx = indices.difference(most_sim_pair).pop()
        choices[s] += (ooo_idx == 2)
        probas[s] += F.softmax(dots * temperature, dim=0)
    return choices, probas


def accuracy(choices: List[bool]) -> float:
    """Computes the odd-one-out triplet accuracy."""
    return round((choices.sum() / choices.shape[0]).item(), 4)


def ventropy(probabilities: Tensor) -> Tensor:
    """Computes the entropy for a batch of (discrete) probability distributions."""
    def entropy(p: Tensor) -> Tensor:
        return -(torch.where(p > torch.tensor(0.), p * torch.log(p), torch.tensor(0.))).sum()

    return vmap(entropy)(probabilities)


def save_triplet_probas(
    probas: Tensor, out_path: str, model_name: str, module_name: str
) -> None:
    """Saves triplet probabilities to disk."""
    out_path = os.path.join(out_path, model_name, module_name)
    if not os.path.exists(out_path):
        print("\nCreating output directory...\n")
        os.makedirs(out_path)
    with open(os.path.join(out_path, "triplet_probas.npy"), "wb") as f:
        np.save(f, probas.cpu().numpy())


def evaluate(args, backend: str = "pt") -> None:
    device = torch.device(args.device)
    model_cfg, data_cfg = create_hyperparam_dicts(args)
    results = []
    for model_name, module_name in tqdm(zip(model_cfg.names, model_cfg.modules)):
        model = Model(
            model_name, pretrained=True, model_path=None, device=device, backend=backend
        )
        things_behavior = THINGSBehavior(
            root=data_cfg.root, 
            transform=model.get_transformations(),
            download=False
            )
        dl = DataLoader(
            dataset=things_behavior,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
            )
        features, _ = model.extract_features(
            data_loader=dl,
            module_name=module_name,
            flatten_acts=True,
            clip=True if re.compile(r"^clip").search(model_name.lower()) else False,
            return_probabilities=False
            )
        triplets = things_behavior.get_triplets()
        choices, probas = get_predictions(features, triplets, args.temperature)
        acc = accuracy(choices)
        entropies = ventropy(probas)
        mean_entropy = entropies.mean().item()
        if args.verbose:
            print(
                f"\nModel: {model_name}, Accuracy: {acc:.4f}, Average triplet entropy: {mean_entropy:.3f}\n"
            )
        summary = {
            "model": model_name,
            "accuracy": acc,
            "mean_entropy": mean_entropy,
            "entropies": entropies.cpu().numpy(),
        }
        results.append(summary)
        save_triplet_probas(probas, args.out_path, model_name, module_name)

    # convert results into Pandas DataFrame
    results = pd.DataFrame(results)

    if not os.path.exists(args.out_path):
        print("\nCreating output directory...\n")
        os.makedirs(args.out_path)

    results.to_csv(os.path.join(args.out_path, "results.csv"))


if __name__ == "__main__":
    # parse arguments and set random seeds
    args = parseargs()
    np.random.seed(args.rnd_seed)
    random.seed(args.rnd_seed)
    torch.manual_seed(args.rnd_seed)
    # set number of threads used by PyTorch if device is CPU
    if re.compile(r'^cpu').search(args.device.lower()):
        torch.set_num_threads(args.num_threads)
    # run evaluation script
    evaluate(args)
