



import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from typing import Dict
from tqdm import tqdm

Array = np.ndarray

def partition_triplets(
    triplets: Array,
    objects: Array,
    k: int,
    rnd_seed: int,
) -> Dict[str, Dict[str, list]]:
    triplet_partition = defaultdict(lambda: defaultdict(list))
    kf = KFold(n_splits=k, random_state=rnd_seed, shuffle=True)
    for k, (train_idx, _) in tqdm(enumerate(kf.split(objects)), desc='Fold'):
        train_objects = objects[train_idx]
        for triplet in triplets:
            splits = list(map(lambda obj: 'train' if obj in train_objects else 'val', triplet))
            if len(set(splits)) == 1:
                triplet_partition[f'fold_{k+1:02d}'][set(splits).pop()].append(triplet.tolist())
    return dict(triplet_partition)