import os
import torch
import numpy as np

Tensor = torch.Tensor
Array = np.ndarray


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, disjoint: bool, train: bool):
        super(TripletDataset, self).__init__()
        self.root = root
        self.disjoint = disjoint

        if self.disjoint:
            # TODO: partition triplets into disjoint object sets
            pass
        else:
            self.triplets = self.load_triplets(
                root=self.root,
                file_name="train_90" if train else "test_10",
            )
        self.n_objects = self.get_num_objects(self.triplets)
        self.identity = torch.eye(self.n_objects)

    @staticmethod
    def get_num_objects(triplets: Tensor) -> int:
        """Get number of unique objects in triplet data."""
        n_objects = torch.max(triplets).item()
        if torch.min(triplets).item() == 0:
            n_objects += 1
        return int(n_objects)

    @staticmethod
    def load_triplets(root: str, file_name: str, suffix: str = ".npy") -> Array:
        """Load triplet data from disk."""
        with open(os.path.join(root, "triplets", file_name + suffix), "rb") as f:
            triplets = np.load(f).astype(int)
        triplets = torch.from_numpy(triplets)
        return triplets

    def encode_as_onehot(self, triplet: Tensor) -> Tensor:
        """Encode triplet of numerical indices as three one-hot-vectors."""
        return self.identity[triplet, :]

    def __getitem__(self, idx: int) -> Tensor:
        object_triplet = self.triplets[idx]
        one_hot_triplet = self.encode_as_onehot(object_triplet)
        return one_hot_triplet

    def __len__(self) -> int:
        return self.triplets.shape[0]
