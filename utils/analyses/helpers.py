#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import warnings
from functools import partial
from typing import List

import numpy as np
import pandas as pd

from .families import Families

Array = np.ndarray


def aggregate_dimensions(concept_embedding: Array, idx_triplet: Array) -> Array:
    """Aggregate the histogram of dimensions across the pair of the two most similar objects."""
    triplet_embedding = concept_embedding[idx_triplet]
    pair_dimensions = triplet_embedding[:-1].mean(axis=0)
    return pair_dimensions


def get_max_dims(concept_embedding: Array, triplets: Array) -> Array:
    """Get most important dimension for the most similar object pair in a triplet."""
    aggregate = partial(aggregate_dimensions, concept_embedding)

    def get_max_dim(triplet: Array) -> Array:
        pair_dimensions = aggregate(triplet)
        return np.argmax(pair_dimensions)

    return np.apply_along_axis(get_max_dim, axis=1, arr=triplets)


def get_topk_dims(concept_embedding: Array, triplets: Array, k: int = 2) -> Array:
    """Get top-k most important dimension for the most similar object pair in a triplet."""
    aggregate = partial(aggregate_dimensions, concept_embedding)

    def get_topks(k: int, triplet: Array) -> Array:
        aggregated_dimensions = aggregate(triplet)
        return np.argsort(-aggregated_dimensions)[:k]

    return np.apply_along_axis(partial(get_topks, k), axis=1, arr=triplets).flatten()


def get_failures(triplets: Array, model_choices: Array, target: int = 2) -> Array:
    """Partition triplets into failure and correctly predicted triplets."""
    model_failures = np.where(model_choices != target)[0]
    failure_triplets = triplets[model_failures]
    return failure_triplets


def get_family_name(model_name: str) -> str:
    families = Families([model_name])
    all_children = [attr for attr in dir(families) if attr.endswith("children")]
    for children in all_children:
        if getattr(families, children):
            family_name = families.mapping[children]
            if not family_name == "CNN" or family_name == "SSL":
                break
    return family_name


def merge_results(
        root: str, model_sources: List[str], dataset: str, layer: str
) -> pd.DataFrame:
    results = []
    for source in model_sources:
        results_path = os.path.join(root, dataset, source, layer)
        try:
            source_results = get_results(results_path)
        except FileNotFoundError:
            warnings.warn(
                f"\nCould not find any results for source: <{source}> and layer: <{layer}>.\n"
            )
            continue
        if "source" not in source_results.columns.values:
            source_results["source"] = source
        results.append(source_results)
    results = pd.concat(results, axis=0, ignore_index=True)
    return results


def get_results(root: str) -> pd.DataFrame:
    return pd.read_pickle(os.path.join(root, "results.pkl"))


def map_model_name(name: str) -> str:
    mapping = {
        "resnet_v1_50_tpu_softmax_seed0": "ResNet50 (Softmax)",
        "resnet_v1_50_tpu_label_smoothing_seed0": "ResNet50 (Label smooth.)",
        "resnet_v1_50_tpu_sigmoid_seed0": "ResNet50 (Sigmoid)",
        "resnet_v1_50_tpu_extra_wd_seed0": "ResNet50 (Extra L2)",
        "resnet_v1_50_tpu_dropout_seed0": "ResNet50 (Dropout)",
        "resnet_v1_50_tpu_logit_penalty_seed0": "ResNet50 (Logit penalty)",
        "resnet_v1_50_tpu_nt_xent_weights_seed0": "ResNet50 (Cosine softmax)",
        "resnet_v1_50_tpu_nt_xent_seed0": "ResNet50 (Logit norm)",
        "resnet_v1_50_tpu_squared_error_seed0": "ResNet50 (Squared error)",
        "resnet_v1_50_tpu_mixup_seed0": "ResNet50 (MixUp)",
        "resnet_v1_50_tpu_autoaugment_seed0": "ResNet50 (AutoAugment)",
        "align": "ALIGN",
        "basic-l": "Basic-L",
        "mobilenet_v2_1": "MobileNet v2 (1.4x)",
        "nasnet_mobile-retrained-no_ls-no_dropout-no_aux": "NASNet Mobile",
        "nasnet_large-retrained-no_ls-no_dropout-no_aux": "NASNet Large",
        "vice": "VICE",
        "Alexnet_ecoset": "AlexNet (Ecoset)",
        "VGG16_ecoset": "VGG-16 (Ecoset)",
        "Resnet50_ecoset": "RN-50 (Ecoset)"
    }

    if name in mapping:
        name = mapping[name].replace("ResNet", "RN")
    else:
        if "retrained" in name:
            name = name.replace("_tpu", "").replace("_keras", "")
            name = name.split("-retrained")[0]
        else:
            name = name.replace("patch", "")
            name = (
                name.replace("_small_", "-S/")
                    .replace("_tiny_", "-T/")
                    .replace("_base_", "-B/")
                    .replace("_large_", "-L/")
            )
            name = name.replace("-i1k", " (i1k)").replace("-i21k", " (i21k)")

        name = (
            name.replace("r50", "RN-50").replace("resnet", "RN-")
        )

        if name.endswith("-rn50"):
            name = "RN-50-"+name[:-5]

        capitalize = {
            "net": "Net",
            "_b": "_B",
            "efficient": "Efficient",
            "vgg": "VGG-",
            "mobile": "Mobile",
            "alex": "Alex",
            "vit": "ViT",
            "incept": "Incept",
            "dense": "Dense",
            "resnext": "ResNeXt",
            "clip": "CLIP",
            "mocov": "MoCoV",
            "simclr": "SimCLR",
            "barlowt": "BarlowT",
            "swav": "Swav",
            "jigsaw": "Jigsaw",
            "vicreg": "VICReg",
            "rot": "Rot",
        }
        for k, v in capitalize.items():
            name = name.replace(k, v)

        name = name.replace(" openai", "")
        name = name.replace("Net_", "Net")
        name = name.replace("_", " ")

        if name == "RN v1 50":
            name = "RN50 v1"
        elif name == "RN v1 101":
            name = "RN101 v1"
        elif name == "RN v1 152":
            name = "RN152 v1"
        else:
            name = name.replace("Inception RN", "Inception-RN")
            name = name.replace("etB", "et B")
            name = name.replace("v1", " v1").replace("v2", " v2").replace("  ", " ")
            name = name.replace("ViT ", "ViT-").replace("ViT_", "ViT-")
            name = (
                name.replace("-B ", "-B/")
                    .replace("-T ", "-T/")
                    .replace("-S ", "-S/")
                    .replace("-L ", "-L/")
                    .replace("-G ", "-G/")
            )

    return name


def map_objective_function(model_name, source, family):
    objective = 'Softmax'
    if source == 'vit_same' or model_name in ['resnet_v1_50_tpu_sigmoid_seed0']:
        objective = 'Sigmoid'
    elif model_name in ['resnet_v1_50_tpu_nt_xent_weights_seed0',
                        'resnet_v1_50_tpu_nt_xent_seed0',
                        'resnet_v1_50_tpu_label_smoothing_seed0',
                        'resnet_v1_50_tpu_logit_penalty_seed0',
                        'resnet_v1_50_tpu_mixup_seed0',
                        'resnet_v1_50_tpu_extra_wd_seed0']:
        objective = 'Softmax+'
    elif family.startswith('SSL'):
        objective = 'Self-sup.'
    elif source == 'loss' and model_name == 'resnet_v1_50_tpu_squared_error_seed0':
        objective = 'Squared error'
    elif family in ['Align', 'Basic', 'CLIP']:
        objective = 'Contrastive (image/text)'
    return objective
