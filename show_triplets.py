import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import argparse
from data import load_dataset, DATASETS

color = 'lime'
plt.rcParams['axes.edgecolor'] = color

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='resources/datasets')
parser.add_argument('--dataset', default='cifar100-coarse', choices=DATASETS)
args = parser.parse_args()

dataset = load_dataset(name=args.dataset, data_dir=args.data_dir)

triplets = dataset.get_triplets()
for k in range(20):
    i1, i2, i3 = triplets[k]
    fig, ax = plt.subplots(1, 3)
    for i, idx in enumerate([i1, i2, i3]):
        img, target = dataset[idx]
        img = img.resize((224, 224))
        ax[i].imshow(img)
        if i == 2:
            ax[i].patch.set_edgecolor(color)
            ax[i].patch.set_linewidth('5')
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        else:
            ax[i].axis('off')

    plt.tight_layout()
    plt.savefig(f'resources/plots/cifar_images/{k}.png')
