#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import os
import urllib

import numpy as np
import pandas as pd

from PIL import Image
from typing import Tuple

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

class THINGSBehavior(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, target_transform=None, download=True):
        super(THINGSBehavior, self).__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        if download:
            concept_file = urllib.request.urlopen(object_concepts_link)
        else:
            concept_file = os.path.join(self.root, 'concepts', 'things_concepts.tsv')

        # load train and test triplets
        train_triplets = self.load_triplets(train=True)
        val_triplets = self.load_triplets(train=False)
        self.triplets = np.vstack((train_triplets, val_triplets))
        
        # load object concept names according to which images have to be sorted
        things_objects = pd.read_csv(concept_file, sep='\t', encoding='utf-8')
        object_names = things_objects['uniqueID'].values
        self.names = list(map(lambda n: n + '.jpg', object_names))

    def load_triplets(self, train: bool = True) -> Array:
        with open(os.path.join(self.root, 'triplets', 'train_90.npy' if train else 'test_10.npy'), 'rb') as f:
            triplets = np.load(f).astype(int)
        return triplets

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        img = os.path.join(self.root, 'images', self.names[idx])
        img = Image.open(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor([idx])

    def __len__(self) -> int:
        return len(self.names)
    
    def get_triplets(self):
        return self.triplets