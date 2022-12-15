import os

from .cifar import CIFAR10Triplet, CIFAR100CoarseTriplet, CIFAR100Triplet
from .multi_arrangement import MultiArrangement
from .free_arrangement import FreeArrangement
from .peterson import Peterson
from .things import THINGSBehavior, THINGSTriplet

DATASETS = [
    "cifar100-coarse",
    "cifar100-fine",
    "cifar10",
    "things",
    "things-aligned",
    "multi-arrangement",
    "free-arrangement",
    "peterson",
]


def load_dataset(name: str, data_dir: str, category=None, stimulus_set=None, transform=None):
    if name == "cifar100-coarse":
        dataset = CIFAR100CoarseTriplet(
            triplet_path=os.path.join(data_dir, "cifar100_coarse_triplets.npy"),
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
    elif name == "things":
        dataset = THINGSBehavior(
            root=data_dir, aligned=False, download=True, transform=transform
        )
    elif name == "things-aligned":
        dataset = THINGSBehavior(
            root=data_dir, aligned=True, download=True, transform=transform
        )
    elif name == "multi-arrangement":
        dataset = MultiArrangement(root=data_dir, transform=transform)
    elif name == "free-arrangement":
        assert isinstance(
            stimulus_set, str
        ), "\nSimilarity judgments for the data from King et al. (2019) were collected for two differen stimulus sets.\nPlease provide the stimulus set for which you want to extract features.\n"
        dataset = FreeArrangement(
            root=data_dir,
            stimulus_set=stimulus_set,
            transform=transform,
        )
    elif name == "peterson":
        assert isinstance(
            category, str
        ), "\nSimilarity judgments for the data from Peterson et al. (2016) were collected for individual categories.\nPlease provide a category name.\n"
        dataset = Peterson(
            root=data_dir,
            category=category,
            transform=transform,
        )
    else:
        raise ValueError("\nUnknown dataset\n")

    return dataset
