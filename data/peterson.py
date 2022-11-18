#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import re
from typing import Dict, List, Union

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

Tensor = torch.Tensor
Array = np.ndarray


class Peterson(torch.utils.data.Dataset):
    def __init__(
        self, root: str, category: str = None, transform=None, target_transform=None
    ) -> None:
        super(Peterson, self).__init__()
        self.root = root
        self.category = category
        self.img_subfolder = "images"
        self.sim_subfolder = "sim_judgements"
        self.transform = transform
        self.order = sorted(
            [
                f.name
                for f in os.scandir(
                    os.path.join(self.root, self.category, self.img_subfolder)
                )
                if re.search(r"(jpg|png)$", f.name)
            ]
        )
        self.rsm = self.load_sim_judgements()

    def load_sim_judgements(self) -> Array:
        try:
            rsm_file = self._find_rsm()
            rsm_struct = loadmat(
                os.path.join(self.root, self.category, self.sim_subfolder, rsm_file)
            )
            key = [k for k in rsm_struct.keys() if re.search("sim", k.lower())].pop()
            rsm = rsm_struct[key]
        except FileNotFoundError:
            rsm = self._compute_rsm()
        return rsm

    def _compute_rsm(self, maximum_rating: int = 10) -> Array:
        sim_judgements = self._load_pickle()
        rsm = np.eye(len(self.order))
        rsm *= maximum_rating
        for stimulus_pair, ratings in sim_judgements.items():
            stimulus_i, stimulus_j = stimulus_pair.split(",")
            i = int(re.search(r"\d+", stimulus_i).group()) - 1
            j = int(re.search(r"\d+", stimulus_j).group()) - 1
            avg_similarity = np.mean(ratings)
            rsm[i, j] += avg_similarity
            rsm[j, i] += avg_similarity
        return rsm

    def _load_pickle(
        self, f_name: str = "fruits_mturkRESULTS_all.pickle"
    ) -> Dict[str, List[float]]:
        with open(
            os.path.join(self.root, self.category, self.sim_subfolder, f_name), "rb"
        ) as f:
            sim_judgements = pickle.load(f)
        return sim_judgements

    def _find_rsm(self, fformat: str = ".mat") -> Union[str, Exception]:
        dir = os.path.join(self.root, self.category, self.sim_subfolder)
        for f in os.scandir(dir):
            if re.search(r"sim", f.name.lower()) and f.name.endswith(fformat):
                return f.name
        raise FileNotFoundError(
            f"\nCould not find representational similarity matrix for {self.category}\n"
        )

    def __getitem__(self, idx: int) -> Tensor:
        img = os.path.join(
            self.root, self.category, self.img_subfolder, self.order[idx]
        )
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.order)

    def get_rsm(self) -> Array:
        return self.rsm
