#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import os
import urllib

import numpy as np
import pandas as pd

from PIL import Image
from typing import Any, Tuple
from dataclasses import dataclass

Array = np.ndarray
Tensor = torch.Tensor

object_concepts_link = "https://raw.githubusercontent.com/ViCCo-Group/THINGSvision/master/thingsvision/data/things_concepts.tsv"


class THINGSTriplet(torch.utils.data.Dataset):

    def __init__(self, root, train=True, transform=None, target_transform=None, download=True):
        super(THINGSTriplet, self).__init__()
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.target = 2
    
<<<<<<< HEAD
<<<<<<< HEAD

=======
=======
>>>>>>> fixed typo
        
>>>>>>> fixed typo
        with open(os.path.join(self.root, 'triplets', 'train_90.npy' if train else 'test_10.npy'), 'rb') as f:
            self.triplets = np.load(f).astype(int)

        if download:
            f = urllib.request.urlopen(object_concepts_link)
        else:
            f = os.path.join(self.root, 'concepts', 'things_concepts.tsv')

        things_objects = pd.read_csv(f, sep='\t', encoding='utf-8')
        object_names = things_objects['uniqueID'].values

        self.names = list(map(lambda n: n + '.jpg', object_names))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        triplet = self.triplets[index]
        images = []
        for idx in triplet:
            img = os.path.join(self.root, 'images', self.names[idx])
            img = Image.open(img)
            if self.transform is not None:
                img = self.transform(img)
            images.append(img)

        if self.target_transform is not None:
            self.target = self.target_transform(self.target)
        return images[0], images[1], images[2], self.target

    def __len__(self) -> int:
        return self.triplets.shape[0]