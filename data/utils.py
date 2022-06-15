from data.cifar100 import CIFAR100Triplet, CIFAR100CoarseTriplet, CIFAR10Triplet
from data.things import THINGSTriplet, THINGS
from torch.utils.data import Subset
import random


def load_dataset(name, data_dir, transform):
    if name == 'cifar100-coarse':
        dataset = CIFAR100CoarseTriplet(root=data_dir, train=True,
                                        download=True, transform=transform,
                                        samples=10000, seed=0)
    elif name == 'cifar100-fine':
        dataset = CIFAR100Triplet(root=data_dir, train=True,
                                  download=True, transform=transform,
                                  samples=10000, seed=0)
    elif name == 'cifar10':
        dataset = CIFAR10Triplet(root=data_dir, train=True,
                                 download=True, transform=transform,
                                 samples=10000, seed=0)
    elif name == 'things':
        dataset = THINGSTriplet(root=data_dir, train=True,
                                download=True, transform=transform)
    elif name == 'things-new':
        dataset = THINGS(root=data_dir, train=True,
                         download=True, transform=transform)
    elif name == 'things-5k':
        dataset = THINGSTriplet(root=data_dir, train=True,
                                download=True, transform=transform)
        k = 5000
        indices = list(range(5000))
        random.seed(0)
        random.shuffle(indices)
        dataset = Subset(dataset, indices=indices[:k])
    else:
        raise ValueError('Unknown dataset')

    return dataset
