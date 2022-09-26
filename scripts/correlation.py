import sys

sys.path.append('.')

import argparse
import pandas as pd
import pathlib
import os
import scipy.stats as stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_data_path', default=None)
    parser.add_argument('--x_metric', default='imagenet_accuracy', choices=['imagenet_accuracy', 'param_count'])
    parser.add_argument('--output', default='resources/final_results/plots/big_plot.pdf')
    parser.add_argument('--legend-loc', default="lower left")
    args = parser.parse_args()

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    paths = ['resources/final_results/things/results.csv',
             'resources/final_results/cifar100-coarse/results.csv',
             'resources/final_results/things/probing_results.csv', ]
    legend_locs = ['lower left', 'upper left']

    for idx, path in enumerate(paths):
        results = pd.read_csv(path)
        final_layer_indices = []
        for name, group in results.groupby('model'):
            if 'seed1' in name or name.startswith('resnet_v1_50_tpu_random_init'):
                continue
            if len(group.index) > 2:
                sources = group.source.values.tolist()
                assert 'vit_best' in sources and 'vit_same' in sources
                group = group[group.source == 'vit_same']
            final_layer_indices.append(group[group.accuracy.max() == group.accuracy].index[0])

        final_layer_results = results.iloc[final_layer_indices]
        df = final_layer_results.merge(networks, on='model')
        print(stats.pearsonr(df.accuracy, df.imagenet_accuracy))
