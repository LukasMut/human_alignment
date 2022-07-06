import pickle
import matplotlib.pyplot as plt
from data import DATASETS
import argparse
import pandas as pd
import pathlib
import os

DATASET_NAMES = {
    'cifar100-coarse': 'CIFAR100 Coarse',
    'cifar10': 'CIFAR10',
    'cifar100-fine': 'CIFAR100',
    'things': 'Things',
}
COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'black', 'brown']


def load_dataframe(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS, default='things')
    parser.add_argument('--path', default='resources/results/things/penultimate/results.pkl')
    parser.add_argument('--network_data_path', default=None)
    parser.add_argument('--x_metric', default='imagenet_accuracy', choices=['imagenet_accuracy', 'param_count'])
    parser.add_argument('--title', default='Logits')
    args = parser.parse_args()

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    dataset = args.dataset
    results_path = args.path

    results = load_dataframe(results_path)
    fig, ax = plt.subplots()
    df = results.merge(networks, on='model')
    for i, group in enumerate(df['arch'].unique()):
        df[df.arch == group].plot(x=args.x_metric, y='accuracy', ax=ax,
                                  label=group, marker='x', kind='scatter',
                                  c=COLORS[i])

    plt.xlabel('#Parameters' if args.x_metric == 'param_count' else 'Imagenet Accuracy')
    plt.ylabel(f'{DATASET_NAMES[dataset]} Accuracy')
    plt.grid()
    plt.title(args.title)
    plt.savefig('result.png')
    plt.show()
