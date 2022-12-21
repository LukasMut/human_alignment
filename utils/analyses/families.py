#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from dataclasses import dataclass
from typing import Dict, List


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
            "clip_children": "CLIP",
            "vit_in_children": "ViT-IN",
            "vit_jft_children": "ViT-JFT",
            "alexnet_children": "AlexNet",
            "resnet_children": "ResNet",
            "vgg_children": "VGG",
            "ssl_children": "SSL",
            "ssl_contrastive_children": "SSL (contrastive)",
            "ssl_non_contrastive_children": "SSL (non-contrastive)",
            "ssl_non_siamese_children": "SSL (non-siamese)",
            "resnext_children": "ResNext",
            "cnn_children": "CNN",
            "efficientnet_children": "EfficientNet",
            "inception_children": "Inception",
            "mobilenet_children": "MobileNet",
            "nasnet_children": "NasNet",
            "densenet_children": "DenseNet",
            "basic_children": "Basic",
            "align_children": "Align",
        }
        return mapping

    @property
    def vit_in_children(self):
        return self.search("vit_in")

    @property
    def vit_jft_children(self):
        return self.search("vit_jft")

    @property
    def inception_children(self):
        return self.search("inception")

    @property
    def align_children(self):
        return self.search("align")

    @property
    def basic_children(self):
        return self.search("basic")

    @property
    def mobilenet_children(self):
        return self.search("mobilenet")

    @property
    def densenet_children(self):
        return self.search("densenet")

    @property
    def nasnet_children(self):
        return self.search("nasnet")

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
    def vit_in(self):
        return r"^(vit_tiny|vit_small|vit_base|vit_large|vit-s|vit-b|vit-l|vit_s|vit_b|vit_l)"

    @property
    def vit_jft(self):
        return r"^vit-g"

    @property
    def ssl(self):
        return r"(-rn50)$"

    @property
    def inception(self):
        return r"^inception"

    @property
    def align(self):
        return r"^align"

    @property
    def basic(self):
        return r"^basic"

    @property
    def densenet(self):
        return r"^densenet"

    @property
    def mobilenet(self):
        return r"^mobilenet"

    @property
    def nasnet(self):
        return r"^nasnet"

    @property
    def ssl_contrastive(self):
        return f"({self.simclr})|({self.movcov})"

    @property
    def ssl_non_contrastive(self):
        return f"({self.swav})|({self.vicreg})|({self.barlowtins})"

    @property
    def ssl_non_siamese(self):
        return f"({self.rotnet})|({self.jigsaw})"

    @property
    def vicreg(self):
        return f"(?=^vicreg)(?=.*{self.ssl}$)"

    @property
    def swav(self):
        return f"(?=^swav)(?=.*{self.ssl}$)"

    @property
    def barlowtins(self):
        return f"(?=^barlowtins)(?=.*{self.ssl}$)"

    @property
    def simclr(self):
        return f"(?=^simclr)(?=.*{self.ssl}$)"

    @property
    def movcov(self):
        return f"(?=^mocov[0-9]+)(?=.*{self.ssl}$)"

    @property
    def jigsaw(self):
        return f"(?=^jigsaw)(?=.*{self.ssl}$)"

    @property
    def rotnet(self):
        return f"(?=^rotnet)(?=.*{self.ssl}$)"

    @property
    def cnn(self):
        return f"({self.alexnet}|{self.vgg}|{self.resnet}|{self.resnext})"

    @property
    def resnet(self):
        return r"^resnet"

    @property
    def vgg(self):
        return "vgg"

    @property
    def resnext(self):
        return "resnext"

    @property
    def alexnet(self):
        return "alexnet"
