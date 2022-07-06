import os

import matplotlib.pyplot as plt
import argparse
import pickle
import pathlib

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='things')
    parser.add_argument('--base-dir', default='results')
    parser.add_argument('--x-metric', default='imagenet_accuracy', choices=['imagenet_accuracy', 'param_count'])
    args = parser.parse_args()

    name_mapping = {
        'r50-barlowtwins': 'Barlow Twins',
        'r50-vicreg': 'VicReg',
        'r50-swav': 'SwAV',
        'r50-simclr': 'SimCLR',
        'r50-mocov2': 'MoCoV2',
        'r50-rotnet': 'RotNet',
        'r50-jigsaw': 'Jigsaw Puzzle',
        'resnet50': 'Supervised',
    }

    dataset = args.dataset
    results_folder = args.base_dir

    dataset_names = {
        'cifar100-coarse': 'CIFAR100 Coarse',
        'cifar10': 'CIFAR10',
        'cifar100-fine': 'CIFAR100',
        'things': 'Things',
    }

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'brown', 'yellow']

    with open(os.path.join(results_folder, 'results.pkl'), 'rb') as f:
        df = pickle.load(f)

    networks = pd.read_csv(os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv'))
    df = df.merge(networks, on='model')

    fig, ax = plt.subplots()

    for i, model in enumerate(df['model'].values):
        df[df.model == model].plot(x='imagenet_accuracy', y='accuracy', ax=ax,
                                   label=name_mapping[model],
                                   marker='x', kind='scatter', c=colors[i])

    plt.xlabel('#Parameters' if args.x_metric == 'param_count' else 'Imagenet Accuracy')
    plt.ylabel(f'{dataset_names[dataset]} Accuracy')
    plt.grid()
    plt.savefig('result.png')
    plt.show()
