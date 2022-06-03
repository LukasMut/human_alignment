import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import argparse
from data.utils import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir')
parser.add_argument('--dataset', default='cifar100', choices=['things', 'cifar100'])
args = parser.parse_args()

dataset = load_dataset(name=args.dataset, data_dir=args.data_dir, transform=transforms.ToTensor())

dl = DataLoader(dataset, batch_size=1)
it = iter(dl)
for _ in range(10):
    x1, x2, x3, y = next(it)
    fig, ax = plt.subplots(1, 3)
    for i, x in enumerate([x1, x2, x3]):
        ax[i].imshow(torch.permute(x[0], (1, 2, 0)))
        ax[i].axis('off')
    plt.show()
