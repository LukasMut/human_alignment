import argparse
import random
import numpy as np
from data.cifar import CIFAR100Coarse
from collections import defaultdict


def generate_triplets(dataset, seed: int, samples: int):
    random.seed(seed)
    labels = np.unique(dataset.targets)
    class_indices = []
    for label in labels:
        class_indices.append(np.argwhere(label == dataset.targets).flatten().tolist())
        random.shuffle(class_indices)
    triplets = []
    for sample in range(samples):
        equal_cls = random.choice(labels)
        other_cls = random.choice(list(filter(lambda x: x != equal_cls, labels)))
        triplet = []
        for _ in range(2):
            idx = random.choice(class_indices[equal_cls])
            # class_indices[equal_cls].remove(idx)
            triplet.append(idx)
        idx = random.choice(class_indices[other_cls])
        # class_indices[other_cls].remove(idx)
        triplet.append(idx)
        triplets.append(triplet)
    triplets = np.array(triplets)
    return triplets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='resources/cifar100_coarse_triplets')
    args = parser.parse_args()

    dataset = CIFAR100Coarse(root='resources/datasets', train=True, download=True)
    triplets = generate_triplets(dataset, seed=args.seed, samples=50000)
    counter = defaultdict(int)
    for t in triplets:
        for idx in t:
            counter[dataset.targets[idx].item()] += 1
    print(counter)

    np.save(args.output, triplets)
