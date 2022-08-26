#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from functools import partial
from typing import Dict, List

import numpy as np
import pandas as pd

from . import helpers
from .families import Families

Array = np.ndarray


@dataclass
class Failures:
    results: pd.DataFrame
    triplet_dataset: object
    concept_embedding: Array
    iv: str
    concept_importance: str = None
    human_entropies: Array = None

    def __post_init__(self):
        self.models = self.results.model.unique()
        self.families = Families(self.models)
        self.triplets = self.triplet_dataset.triplets
        self.classification_errors = dict()
        self.n_families = 0

        assert self.iv in ["dimension", "entropy"]
        if self.iv == "dimension":
            assert self.concept_importance in ["max", "topk"]
            importance_fun = getattr(helpers, f"get_{self.concept_importance}_dims")
            self.importance_fun = partial(importance_fun, self.concept_embedding)
            self.n_triplets_per_bin = self.get_triplets_per_bin(self.triplets)
            self.n_bins = self.concept_embedding.shape[-1]
        else:  # entropy
            self.boundaries = np.arange(0, np.log(3) + 1e-1, 1e-1)
            assert isinstance(
                self.human_entropies, np.ndarray
            ), "\nVICE entropies required to compute zero-one loss per entropy bucket.\n"
            self.triplet_assignments = np.digitize(
                self.human_entropies, bins=self.boundaries, right=True
            )
            self.n_triplets_per_bin = self.get_triplets_per_bin(
                np.arange(self.triplets.shape[0])
            )
            self.n_bins = self.boundaries.shape[0] - 1

    def get_model_subset(self, family: str) -> List[str]:
        return getattr(self.families, family)

    def get_failures(self, model_choices: Array) -> Array:
        """Partition triplets into failure and correctly predicted triplets."""
        model_failures = np.where(model_choices != 2)[0]
        if self.iv == "dimension":
            model_failures = self.triplets[model_failures]
        return model_failures

    def get_triplets_per_bin(self, triplets: Array) -> Array:
        if self.iv == "dimension":
            triplet_assignments = self.importance_fun(triplets)
        else:  # entropy
            triplet_assignments = self.triplet_assignments[triplets]
        num_triplets_per_bin = np.bincount(triplet_assignments)[
            triplet_assignments.min() :
        ]
        return num_triplets_per_bin

    def compute_classification_errors(self, family: str) -> None:
        children = self.get_model_subset(family)
        family_subset = self.results[self.results.model.isin(children)]
        family_subset.reset_index(drop=True, inplace=True)
        classification_errors = np.zeros((family_subset.shape[0], self.n_bins))
        for i, child_data in family_subset.iterrows():
            model_choices = child_data["choices"]
            # get triplet indices for which a model predictly differently than humans
            failure_triplets = self.get_failures(model_choices)
            n_failures_per_bin = self.get_triplets_per_bin(failure_triplets)
            binwise_zero_one_loss = n_failures_per_bin / self.n_triplets_per_bin
            classification_errors[i] += binwise_zero_one_loss

        # average zero-one losses per dimesions over the children of a family
        family_classification_errors = classification_errors.mean(axis=0)
        self.classification_errors.update(
            {self.families.mapping[family]: family_classification_errors}
        )

    def update(self, family: str) -> None:
        self.compute_classification_errors(family)
        self.n_families += 1

    @property
    def family_zero_one_losses(self) -> Dict[str, Array]:
        return self.classification_errors