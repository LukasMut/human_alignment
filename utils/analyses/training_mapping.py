from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class Mapper:
    results: pd.DataFrame

    def __post_init__(self) -> None:
        self.models = self.results.model.values
        self.families = self.results.family.values
        self.sources = self.results.source.values
        self.objectives = ["imagenet1k", "imagenet21k", "jft30k", "imagetext"]

    def get_training_objectives(self) -> List[str]:
        training_objectives = [
            self.check_conditions(model, family, source)
            for (model, family, source) in zip(self.models, self.families, self.sources)
        ]
        return training_objectives

    def imagenet1k_condition(self, model: str, family: str, source: str) -> str:
        if (
            not self._is_clip_model(family)
            and not self._is_source_google(source)
            and not self._is_ssl_model(family)
            and not self._is_imagenet21k_model(model)
        ):
            return self.imagenet1k_objective

    def imagenet21k_condition(self, model: str) -> str:
        if self._is_imagenet21k_model(model):
            return self.imagenet21k_objective

    def jft30k_condition(self, family: str, source: str) -> str:
        if self._is_jft30k_model(family, source):
            return self.jft30k_objective

    def imagetext_condition(self, family: str) -> str:
        if self._is_imagetext_model(family):
            return self.imagetext_objective

    def check_conditions(self, model: str, family: str, source: str) -> str:
        for objective in self.objectives:
            if objective == "imagenet1k":
                training = getattr(self, f"{objective}_condition")(
                    model=model, family=family, source=source
                )
            elif objective == "imagenet21k":
                training = getattr(self, f"{objective}_condition")(model)
            elif objective == "jft30k":
                training = getattr(self, f"{objective}_condition")(
                    family=family, source=source
                )
            else:
                training = getattr(self, f"{objective}_condition")(family)
            if training:
                return training
        assert self._is_ssl_model(
            family
        ), f"\nMapping from model, family, and source to training objective did not work correctly for model: <{model}> and source: <{source}.\n"
        return family

    @property
    def imagenet1k_objective(self) -> str:
        return "Supervised (ImageNet 1k)"

    @property
    def imagenet21k_objective(self) -> str:
        return "Supervised (ImageNet 21k)"

    @property
    def jft30k_objective(self) -> str:
        return "Supervised (JFT 30k)"

    @property
    def imagetext_objective(self) -> str:
        return "Image/Text"

    @staticmethod
    def _is_imagenet21k_model(model: str) -> bool:
        return model.endswith("21k")

    @staticmethod
    def _is_clip_model(family: str) -> bool:
        return family == "CLIP"

    @staticmethod
    def _is_ssl_model(family: str) -> bool:
        return family.startswith("SSL")

    @staticmethod
    def _is_source_google(source: str) -> bool:
        return source == "google"

    @staticmethod
    def _is_jft30k_model(family: str, source: str) -> bool:
        return family == "ViT" and source == "google"

    @staticmethod
    def _is_imagetext_model(family: str) -> bool:
        return family in ["CLIP", "Align", "Basic"]
