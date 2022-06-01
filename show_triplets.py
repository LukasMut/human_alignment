from data.cifar100 import CIFAR100Triplet
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

dataset = CIFAR100Triplet(root='resources/datasets', train=True,
                          download=True, transform=transforms.ToTensor(), samples=10000, seed=0)
dl = DataLoader(dataset, batch_size=1)
it = iter(dl)
for _ in range(10):
    x1, x2, x3, y = next(it)
    fig, ax = plt.subplots(1, 3)
    for i, x in enumerate([x1, x2, x3]):
        ax[i].imshow(torch.permute(x[0], (1, 2, 0)))
        ax[i].axis('off')
    plt.show()
