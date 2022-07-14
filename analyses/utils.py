#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from functools import partial

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
