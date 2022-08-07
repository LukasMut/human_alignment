import os
import torch
import numpy as np
import pickle

Tensor = torch.Tensor
Array = np.ndarray


class TripletData(torch.utils.data.Dataset):
    def __init__(self, root: str, n_objects: int, split: str, train: bool, fold=None):
        super(TripletData, self).__init__()
        self.root = root
        self.split = split
        self.train = train
        self.identity = torch.eye(n_objects)

        if self.split == "disjoint":
            assert isinstance(
                fold, str
            ), "\nFor triplet data split with disjoint objects, number of folds is required.\n"
            disjoint_triplets = self.load_disjoint_triplets(
                root=self.root, fname="disjoint_triplets"
            )
            self.triplets = disjoint_triplets[fold]
            if self.train:
                self.triplets = torch.tensor(self.triplets["train"])
            else:
                self.triplets = torch.tensor(self.triplets["val"])
        else:
            self.triplets = self.load_original_triplets(
                root=self.root,
                fname="train_90" if train else "test_10",
            )

    def load_disjoint_triplets(self, root: str, fname: str, suffix: str = ".pkl"):
        with open(
            os.path.join(root, "triplets", self.split, fname + suffix), "rb"
        ) as f:
            disjoint_triplets = pickle.load(f)
        return disjoint_triplets

    def load_original_triplets(
        self, root: str, fname: str, suffix: str = ".npy"
    ) -> Array:
        with open(
            os.path.join(root, "triplets", self.split, fname + suffix), "rb"
        ) as f:
            triplets = np.load(f).astype(int)
        triplets = torch.from_numpy(triplets)
        return triplets

    def encode_as_onehot(self, triplet: Tensor) -> Tensor:
        """encode triplet of objects as three one-hot-vectors"""
        return self.identity[triplet, :]

    def __getitem__(self, index: int) -> Tensor:
        object_triplet = self.triplets[index]
        one_hot_triplet = self.encode_as_onehot(object_triplet)
        return one_hot_triplet

    def __len__(self) -> int:
        return self.triplets.shape[0]
