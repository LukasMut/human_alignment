from data.cifar100 import CIFAR100Triplet
from data.things import THINGSTriplet


def load_dataset(name, data_dir, transform):
    if name == 'cifar100':
        dataset = CIFAR100Triplet(root=data_dir, train=True,
                                  download=True, transform=transform,
                                  samples=10000, seed=0)
    elif name == 'things':
        dataset = THINGSTriplet(root=data_dir, train=True,
                                download=True, transform=transform)
    else:
        raise ValueError('Unknown dataset')

    return dataset
