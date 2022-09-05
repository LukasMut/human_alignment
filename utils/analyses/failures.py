#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import defaultdict
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
    triplets: Array
    concept_embedding: Array
    iv: str
    concept_importance: str = None
    human_entropies: Array = None

    def __post_init__(self):
        self.models = self.results.model.unique()
        self.families = Families(self.models)
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

    def get_correct_predictions(self, model_choices: Array) -> Array:
        """Partition triplets into failure and correctly predicted triplets."""
        correct_predictions = np.where(model_choices == 2)[0]
        if self.iv == "dimension":
            correct_predictions = self.triplets[correct_predictions]
        return correct_predictions

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
        classification_errors = defaultdict(dict)
        for _, child_data in family_subset.iterrows():
            model_choices = child_data["choices"]
            # get triplet indices for which a model predictly differently than humans
            correct_predictions = self.get_correct_predictions(model_choices)
            num_hits_per_bin = self.get_triplets_per_bin(correct_predictions)
            binwise_zero_one_loss = 1 - (num_hits_per_bin / self.n_triplets_per_bin)
            classification_errors[child_data.model][
                child_data.source
            ] = binwise_zero_one_loss.tolist()

        # average zero-one losses per dimesions over the children of a family
        family_classification_errors = [
            zero_one_loss
            for sources in classification_errors.values()
            for zero_one_loss in sources.values()
        ]
        family_classification_errors = np.stack(family_classification_errors, axis=0)
        family_classification_error = family_classification_errors.mean(axis=0)

        classification_errors.update({"overall": {"all": family_classification_error}})
        self.classification_errors.update(
            {self.families.mapping[family]: classification_errors}
        )

    def update(self, family: str) -> None:
        self.compute_classification_errors(family)
        self.n_families += 1

    @property
    def family_zero_one_losses(self) -> Dict[str, Array]:
        return self.classification_errors
