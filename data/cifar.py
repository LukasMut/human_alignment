import numpy as np
from torchvision.datasets import CIFAR100, CIFAR10
import random


class CIFAR100Coarse(CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100Coarse, self).__init__(root, train, transform, target_transform, download)

        # update labels
        coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                                  3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                                  6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                                  0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                                  5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                                  16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                                  10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                                  2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                                  16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                                  18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
        self.targets = coarse_labels[self.targets]

        # update classes
        self.classes = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        ['baby', 'boy', 'girl', 'man', 'woman'],
                        ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]


class RandomMatchingMixin:
    """Randomly creates triplets by matching samples from the same class with one sample from a different class """

    def setup(self, seed: int, samples: int):
        random.seed(seed)
        labels = np.unique(self.targets)
        class_indices = []
        for label in labels:
            class_indices.append(np.argwhere(label == self.targets).flatten().tolist())
            random.shuffle(class_indices)
        self.triplets = []
        for sample in range(samples):
            equal_cls = random.choice(labels)
            other_cls = random.choice(list(filter(lambda x: x != equal_cls, labels)))
            triplet = []
            for _ in range(2):
                idx = random.choice(class_indices[equal_cls])
                class_indices[equal_cls].remove(idx)
                triplet.append(idx)
            idx = random.choice(class_indices[other_cls])
            class_indices[other_cls].remove(idx)
            triplet.append(idx)
            self.triplets.append(triplet)
        self.triplets = np.array(self.triplets)

    def get_triplets(self):
        return self.triplets


class CIFAR100CoarseTriplet(CIFAR100Coarse, RandomMatchingMixin):
    def __init__(self, root, seed=0, samples=10000, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.setup(seed=seed, samples=samples)


class CIFAR100Triplet(CIFAR100, RandomMatchingMixin):
    def __init__(self, root, seed=0, samples=10000, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.setup(seed=seed, samples=samples)


class CIFAR10Triplet(CIFAR10, RandomMatchingMixin):
    def __init__(self, root, seed=0, samples=10000, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.setup(seed=seed, samples=samples)
