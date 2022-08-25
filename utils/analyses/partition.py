#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from dataclasses import dataclass
from functools import partial
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import helpers
from .families import Families

Array = np.ndarray


@dataclass
class Partition:
    """Class to parition results into different subsets."""

    results: pd.DataFrame
    triplet_dataset: object
    concept_importance: str
    concept_embedding: Array
    family_i: str
    family_j: str
    target: int = 2

    def __post_init__(self):
        assert self.concept_importance in ["max", "topk"]
        importance_fun = getattr(helpers, f"get_{self.concept_importance}_dims")
        self.importance_fun = partial(importance_fun, self.concept_embedding)
        self.models = self.results.model.unique()
        self.families = Families(self.models)
        self.triplets = self.triplet_dataset.triplets
        self.triplet_dimensions = self.importance_fun(self.triplets)
        self.familywise_failure_differences = self.get_failure_differences()
        self.familywise_hit_failure_intersection = self.get_hit_failure_intersection()

    def get_model_subset(self, family: str) -> List[str]:
        return getattr(self.families, family)

    def get_children_columns(self, family_failures, family):
        return [
            family_failures.columns.tolist().index(model)
            for model in self.get_model_subset(family)
        ]

    @staticmethod
    def convert_choices(probas: Array) -> Array:
        """Labels for cross-entropy and clasification error are rotations of each other."""
        pair_choices = probas.argmax(axis=1)
        firt_conversion = np.where(pair_choices != 1, pair_choices - 2, pair_choices)
        ooo_choices = np.where(firt_conversion < 0, 2, firt_conversion)
        return ooo_choices

    def get_model_choices(self) -> Array:
        """Get the odd-one-out choices for every triplet for every model."""
        """
        model_choices = np.stack(
            [
                self.results[self.results.model == model]
                .probas.apply(self.convert_choices)
                .values[0]
                for model in self.models
            ],
            axis=1,
        )
        """
        model_choices = np.stack(
            [
                self.results[self.results.model == model].choices.values[0]
                for model in self.models
            ],
            axis=1,
        )

        return model_choices

    def get_children_choices(self, model_subset):
        """Compute the choices of the children belonging to a family."""
        model_choices = self.get_model_choices()
        children_choices = model_choices[:, self.results.model.isin(model_subset)]
        return children_choices

    @property
    def family_i_hits(self) -> Array:
        model_subset = self.get_model_subset(self.family_i)
        children_family_i_choices = self.get_children_choices(model_subset)
        family_i_hits = np.where(
            np.sum(children_family_i_choices, axis=1)
            == (self.target * len(model_subset))
        )[0]
        return family_i_hits

    def filter_failures(self, model_choices: Array) -> Tuple[List[int], Array]:
        failures, choices = zip(
            *list(filter(lambda kv: self.target not in kv[1], enumerate(model_choices)))
        )
        return failures, np.asarray(choices)

    @staticmethod
    def get_intersection(family_failures: pd.DataFrame) -> Array:
        """Find the intersection of failures between the children belonging to a family."""

        def equality_condition(children_choices: Array) -> bool:
            def is_equal(children_choices: Array) -> bool:
                return np.unique(children_choices).shape[0] == 1

            return is_equal(children_choices)

        return np.apply_along_axis(equality_condition, axis=1, arr=family_failures)

    def get_family_failures(self, family: str) -> pd.DataFrame:
        """Get examples for which the children belonging to a family responded differently than humans."""
        model_subset = self.get_model_subset(family)
        children_choices = self.get_children_choices(model_subset)
        failures, choices = self.filter_failures(children_choices)
        children_failures = pd.DataFrame(
            data=choices, index=failures, columns=model_subset
        )
        return children_failures

    @property
    def family_j_failures(self) -> pd.DataFrame:
        family_failures = self.get_family_failures(self.family_j)
        intersection = self.get_intersection(family_failures)
        examples = family_failures.index[np.where(intersection == True)[0]].to_numpy()
        failures = family_failures.filter(items=examples, axis=0)
        # aggregate choices (take the set over family choices)
        failures = failures.apply(lambda x: x.unique()[0], axis="columns")
        return failures

    @staticmethod
    def _is_different(family_i_choices: Array, family_j_choices: Array) -> bool:
        return np.all(
            [
                family_i_choices.shape[0] == 1,
                family_j_choices.shape[0] == 1,
                np.all(family_i_choices != family_j_choices),
            ]
        )

    def get_differences(self, model_failures: Array) -> Array:
        children_i_cols = self.get_children_columns(model_failures, self.family_i)
        children_j_cols = self.get_children_columns(model_failures, self.family_j)

        def check_choice_difference(model_failure: Array) -> bool:
            family_i_choices = np.unique(model_failure[children_i_cols])
            family_j_choices = np.unique(model_failure[children_j_cols])
            return self._is_different(family_i_choices, family_j_choices)

        return np.apply_along_axis(check_choice_difference, axis=1, arr=model_failures)

    def get_failure_differences(self) -> pd.DataFrame:
        model_failures = self.get_family_failures("models")
        failure_differences = self.get_differences(model_failures)
        family_types = [
            random.choices(self.get_model_subset(self.family_i)).pop(),
            random.choices(self.get_model_subset(self.family_j)).pop(),
        ]
        difference_triplets = model_failures.index[
            np.where(failure_differences == True)[0]
        ].to_numpy()
        failure_differences = model_failures.loc[difference_triplets, family_types]
        failure_differences = failure_differences.rename(
            columns={
                family_types[0]: self.families.mapping[self.family_i],
                family_types[1]: self.families.mapping[self.family_j],
            }
        )
        return failure_differences

    def get_hit_failure_intersection(self) -> pd.DataFrame:
        """
        Find the intersection of triplets for which the children of family i were aligned with humans,
        but not the children of family j.
        """
        children_family_i_hits = self.family_i_hits
        children_family_j_failures = self.family_j_failures
        intersection = np.array(
            list(
                set(children_family_i_hits).intersection(
                    set(children_family_j_failures.index.values)
                )
            )
        )
        hit_failure_intersection = children_family_j_failures[intersection].to_frame(
            self.families.mapping[self.family_j]
        )
        hit_failure_intersection[self.families.mapping[self.family_i]] = np.full_like(
            intersection, self.target, dtype=int
        )
        return hit_failure_intersection

    def dimwise_hit_failure_intersection(self, dimension: int) -> pd.DataFrame:
        return self.familywise_hit_failure_intersection.filter(
            items=np.where(self.triplet_dimensions == dimension)[0], axis=0
        )

    def dimwise_failure_differences(self, dimension: int) -> pd.DataFrame:
        return self.familywise_failure_differences.filter(
            items=np.where(self.triplet_dimensions == dimension)[0], axis=0
        )
