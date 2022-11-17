from .things import THINGSBehavior
from .cifar import CIFAR100Triplet, CIFAR100CoarseTriplet, CIFAR10Triplet
from .things import THINGSTriplet, THINGSBehavior
from .multi_arrangement import MultiArrangement
import os

DATASETS = ['cifar100-coarse', 'cifar100-fine', 'cifar10', 'things', 'things-aligned', 'multi-arrangement']


def load_dataset(name: str, data_dir: str, transform=None):
    if name == 'cifar100-coarse':
        dataset = CIFAR100CoarseTriplet(
            triplet_path=os.path.join(data_dir, 'cifar100_coarse_triplets.npy'),
            root=data_dir, train=True,
            download=True, transform=transform)
    elif name == 'things':
        dataset = THINGSBehavior(
            root=data_dir, aligned=False,
            download=True, transform=transform)
    elif name == 'things-aligned':
        dataset = THINGSBehavior(
            root=data_dir, aligned=True,
            download=True, transform=transform)
    elif name == 'multi-arrangement':
        dataset = MultiArrangement(
            root=data_dir, transform=transform
        )
    else:
        raise ValueError('\nUnknown dataset\n')

    return dataset
