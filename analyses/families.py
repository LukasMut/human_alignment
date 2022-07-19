#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import Dict, List

import re


@dataclass
class Families:
    models: List[str]

    def search(self, family: str) -> List[str]:
        children = [
            model
            for model in self.models
            if re.compile(getattr(self, family)).search(model.lower())
        ]
        return children

    @property
    def mapping(self) -> Dict[str, str]:
        mapping = {
            "clip_children": "CLIP-models",
            "vit_children": "ViTs",
            "alexnet_children": "AlexNet",
            "resnet_children": "ResNets",
            "vgg_children": "VGGs",
            "ssl_children": "SSLs",
            "ssl_contrastive_children": "SSL contrastive",
            "ssl_non_contrastive_children": "SSL non-contrastive",
            "ssl_non_siamese_children": "SSL non-siamese",
            "resnext_children": "ResNexts",
            "cnn_children": "CNNs",
            "efficientnet_children": "EfficientNets",
        }
        return mapping

    @property
    def vit_children(self):
        return self.search("vit")

    @property
    def clip_children(self):
        return self.search("clip")

    @property
    def cnn_children(self):
        return self.search("cnn")

    @property
    def ssl_children(self):
        return self.search("ssl")

    @property
    def ssl_contrastive_children(self):
        return self.search("ssl_contrastive")

    @property
    def ssl_non_contrastive_children(self):
        return self.search("ssl_non_contrastive")

    @property
    def ssl_non_siamese_children(self):
        return self.search("ssl_non_siamese")

    @property
    def alexnet_children(self):
        return self.search("alexnet")

    @property
    def vgg_children(self):
        return self.search("vgg")

    @property
    def resnet_children(self):
        return self.search("resnet")

    @property
    def resnext_children(self):
        return self.search("resnext")

    @property
    def efficientnet_children(self):
        return self.search("efficientnet")

    @property
    def efficientnet(self):
        return "efficientnet"

    @property
    def clip(self):
        return "clip"

    @property
    def vit(self):
        # NOTE: do we want to include CLIP-ViT in the set of ViTs or not?
        return r"^vit"  # 'vit'

    @property
    def ssl(self):
        return r"^r50"

    @property
    def ssl_contrastive(self):
        return f"({self.simclr})|({self.movcov}))"

    @property
    def ssl_non_contrastive(self):
        return f"({self.swav})|({self.vicreg})|({self.barlowtins}))"

    @property
    def ssl_non_siamese(self):
        return f"({self.rotnet})|({self.jigsaw})"

    @property
    def vicreg(self):
        return f"(?=^{self.ssl})(?=.*vicreg$)"

    @property
    def swav(self):
        return f"(?=^{self.ssl})(?=.*swav$)"

    @property
    def barlowtins(self):
        return f"(?=^{self.ssl})(?=.*barlowtins$)"

    @property
    def simclr(self):
        return f"(?=^{self.ssl})(?=.*simclr$)"

    @property
    def movcov(self):
        return f"(?=^{self.ssl})(?=.*mocov[0-9]+$)"

    @property
    def jigsaw(self):
        return f"(?=^{self.ssl})(?=.*jigsaw$)"

    @property
    def rotnet(self):
        return f"(?=^{self.ssl})(?=.*rotnet$)"


    @property
    def cnn(self):
        return f"({self.alexnet}|{self.vgg}|{self.resnet}|{self.resnext})"

    @property
    def resnet(self):
        return "resnet"

    @property
    def vgg(self):
        return "vgg"

    @property
    def resnext(self):
        return "resnext"

    @property
    def alexnet(self):
        return "alexnet"
