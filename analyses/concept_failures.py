
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List, Tuple
from dataclasses import dataclass
from functools import partial
from . import Families

import numpy as np
import pandas as pd

Array = np.ndarray

@dataclass
class ConceptFailures:
    results: pd.DataFrame
    triplet_dataset: object
    concept_embedding: Array
    concept_importance: str
    target: int = 2
    
    def __post_init__(self):
        assert self.concept_importance in ['max', 'topk']
        self.importance_fun = getattr(
            self, f'get_{self.concept_importance}_dims')
        self.models = self.results.model.unique()
        self.families = Families(self.models)
        self.failure_hit_ratios = dict()
        
    def get_model_subset(self, family: str) -> List[str]:
        return getattr(self.families, family)
        
    def aggregate_dimensions(self, idx_triplet: Array):
        """Aggregate the histogram of dimensions across the pair of the two most similar objects."""
        triplet_embedding = self.concept_embedding[idx_triplet]
        pair_dimensions = triplet_embedding[:-1].mean(axis=0)
        return pair_dimensions
    
    def get_max_dims(self, triplets: Array) -> Array:
        """Get most important dimension for the most similar object pair in a triplet."""
        aggregate = self.aggregate_dimensions
        def get_max_dim(triplet: Array) -> Array:
            pair_dimensions = aggregate(triplet)
            return np.argmax(pair_dimensions)
        return np.apply_along_axis(get_max_dim, axis=1, arr=triplets)

    def get_topk_dims(self, triplets: Array, k: int = 3) -> Array:
        """Get top-k most important dimension for the most similar object pair in a triplet."""
        aggregate = self.aggregate_dimensions
        def get_topks(k: int, triplet: Array) -> Array:
            aggregated_dimensions = aggregate(triplet)
            return np.argsort(-aggregated_dimensions)[:k]
        return np.apply_along_axis(partial(get_topks, k), axis=1, arr=triplets).ravel()
    
    def partition_triplets(self, model_choices: Array) -> Tuple[Array, Array]:
        """Partition triplets into failure and correctly predicted triplets."""
        correct_choices = np.where(model_choices == 1)[0]
        model_failures = np.where(model_choices == 0)[0]
        failure_triplets = self.triplet_dataset.triplets[model_failures]
        correct_triplets = self.triplet_dataset.triplets[correct_choices]
        return failure_triplets, correct_triplets

    def compute_failure_hit_ratio(self, family: str) -> None:
        """Compute the failure-hit ratio per family"""
        children = self.get_model_subset(family)
        family_subset = self.results[self.results.model.isin(children)]
        family_subset.reset_index(drop=True, inplace=True)
        failure_hit_ratios = np.zeros((family_subset.shape[0], self.concept_embedding.shape[1]))
        for i, child_data in family_subset.iterrows():
            model_choices = child_data['choices']
            # partition triplets into "correct" and "failure" triplets based on choices of model
            failure_triplets, correct_triplets = self.partition_triplets(model_choices)
            # get dimensions for which a model correctly predicted or failed to predict the odd-one-out
            correct_dims = self.importance_fun(
                correct_triplets)
            failure_dims = self.importance_fun(
                failure_triplets)
            # get hits per object concept / dimension
            num_corrects_per_dim = np.bincount(correct_dims)
            # get failures per object concept / dimension
            num_failures_per_dim = np.bincount(failure_dims)
            # compute failure-hit ratio
            failure_hit_ratio = (num_failures_per_dim / num_corrects_per_dim)
            failure_hit_ratios[i] += failure_hit_ratio
        # average failure-hit ratio over the children of a family
        family_failure_hit_ratio = failure_hit_ratios.mean(axis=0)
        self.failure_hit_ratios.update({self.families.mapping[family]:family_failure_hit_ratio})
        
    @property
    def failure_hit_ratios(self):
        return self.failure_hit_ratios