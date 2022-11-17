#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

Tensor = torch.Tensor
Array = np.ndarray


class MultiArrangement(torch.utils.data.Dataset):
    def __init__(self, root: str, transform=None, target_transform=None) -> None:
        super(MultiArrangement, self).__init__()
        self.root = root
        self.img_subfolder = "images"
        self.sim_subfolder = "sim_judgements"
        self.transform = transform
        self.target_transform = target_transform
        self.order = sorted(
            [
                f.name
                for f in os.scandir(os.path.join(self.root, self.img_subfolder))
                if f.name.endswith("jpg")
            ]
        )
        self.rdm = self.load_sim_judgements()

    def load_sim_judgements(self) -> Array:
        sim_judgements = loadmat(
            os.path.join(self.root, self.sim_subfolder, "judgments_general.mat")
        )
        rdms = sim_judgements["judgments"]
        # we can simply average across all participants' RDMs
        rdm = rdms.mean(axis=0)
        return rdm

    def __getitem__(self, idx: int) -> Tensor:
        img = os.path.join(self.root, self.img_subfolder, self.order[idx])
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.order)

    def get_rdm(self) -> Array:
        return self.rdm
