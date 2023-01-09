from dataclasses import dataclass
from typing import Dict, List

import pandas as pd


@dataclass
class Mapper:
    results: pd.DataFrame

    def __post_init__(self) -> None:
        self.models = self.results.model.values
        self.families = self.results.family.values
        self.sources = self.results.source.values
        self.objectives = ["ecoset", "imagenet1k", "imagenet21k", "jft30k", "imagetext"]

    def get_training_objectives(self) -> List[str]:
        training_objectives = [
            self.check_conditions(model, family, source)
            for (model, family, source) in zip(self.models, self.families, self.sources)
        ]
        return training_objectives

    def imagenet1k_condition(self, meta_info: Dict[str, str]) -> str:
        if (
            not self._is_clip_model(meta_info)
            and not self._is_source_google(meta_info)
            and not self._is_ssl_model(meta_info)
            and not self._is_imagenet21k_model(meta_info)
            and not self._is_ecoset_model(meta_info)
        ):
            return self.imagenet1k_objective

    def ecoset_condition(self, meta_info: Dict[str, str]) -> str:
        if self._is_ecoset_model(meta_info):
            return self.ecoset_objective

    def imagenet21k_condition(self, meta_info: Dict[str, str]) -> str:
        if self._is_imagenet21k_model(meta_info):
            return self.imagenet21k_objective

    def jft30k_condition(self, meta_info: Dict[str, str]) -> str:
        if self._is_jft30k_model(meta_info):
            return self.jft30k_objective

    def imagetext_condition(self, meta_info: Dict[str, str]) -> str:
        if self._is_imagetext_model(meta_info):
            return self.imagetext_objective

    def check_conditions(self, model: str, family: str, source: str) -> str:
        meta_info = {"model": model, "family": family, "source": source}
        for objective in self.objectives:
            training = getattr(self, f"{objective}_condition")(meta_info)
            if training:
                return training
        assert self._is_ssl_model(
            meta_info
        ), f"\nMapping from model, family, and source to training objective did not work correctly for model: <{model}> and source: <{source}.\n"
        training = "Self-Supervised"
        return training

    @property
    def ecoset_objective(self) -> str:
        return "Supervised (Ecoset)"

    @property
    def imagenet1k_objective(self) -> str:
        return "Supervised (ImageNet-1K)"

    @property
    def imagenet21k_objective(self) -> str:
        return "Supervised (ImageNet-21K)"

    @property
    def jft30k_objective(self) -> str:
        return "Supervised (JFT-3B)"

    @property
    def imagetext_objective(self) -> str:
        return "Image/Text"

    @staticmethod
    def _is_ecoset_model(meta_info: Dict[str, str]) -> bool:
        return meta_info["model"].endswith("ecoset")

    @staticmethod
    def _is_imagenet21k_model(meta_info: Dict[str, str]) -> bool:
        return meta_info["model"].endswith("21k")

    @staticmethod
    def _is_clip_model(meta_info: Dict[str, str]) -> bool:
        return meta_info["family"] == "CLIP"

    @staticmethod
    def _is_ssl_model(meta_info: Dict[str, str]) -> bool:
        return meta_info["family"].startswith("SSL")

    @staticmethod
    def _is_source_google(meta_info: Dict[str, str]) -> bool:
        return meta_info["source"] == "google"

    @staticmethod
    def _is_jft30k_model(meta_info: Dict[str, str]) -> bool:
        return meta_info["family"].startswith("ViT") and meta_info["source"] == "google"

    @staticmethod
    def _is_imagetext_model(meta_info: Dict[str, str]) -> bool:
        return meta_info["family"] in ["CLIP", "Align", "Basic"]
