import sys

sys.path.append('.')

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pathlib
import os
import seaborn as sns
from utils.plotting import PALETTE
from sklearn import datasets, linear_model
import numpy as np

LEGEND_FONT_SIZE = 24
X_AXIS_FONT_SIZE = 35
Y_AXIS_FONT_SIZE = 35
TICKS_LABELSIZE = 25
MARKER_SIZE = 400
AXIS_LABELPAD = 25
xlabel = "ImageNet accuracy"
DEFAULT_SCATTER_PARAMS = dict(s=MARKER_SIZE,
                              alpha=0.9,
                              legend="full")

x_lim = [0.45, 0.92]
y_lim = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_data_path', default=None)
    parser.add_argument('--x_metric', default='imagenet_accuracy', choices=['imagenet_accuracy', 'param_count'])
    parser.add_argument('--output', default='resources/final_results/plots/big_plot.pdf')
    parser.add_argument('--legend-loc', default="lower left")
    parser.add_argument('--paths', nargs='+', default=['resources/final_results/things/results.csv',
                                                       'resources/final_results/cifar100-coarse/results.csv'])
    parser.add_argument('--ylabels', nargs='+',
                        default=['Zero-shot odd-one-out accuracy', ''])
    args = parser.parse_args()

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    legend_locs = ['lower left', 'upper left']

    sns.set_context("paper")
    f = plt.figure(figsize=(28, 10), dpi=200)
    gs = f.add_gridspec(1, 2)
    sns.set_context("talk")
    with sns.axes_style("ticks"):
        for idx, path in enumerate(args.paths):
            f.add_subplot(gs[0, idx])
            results = pd.read_csv(path)
            final_layer_indices = []
            for name, group in results.groupby('model'):
                if len(group.index) > 2:
                    sources = group.source.values.tolist()
                    assert 'vit_best' in sources and 'vit_same' in sources
                    group = group[group.source == 'vit_same']
                final_layer_indices.append(group[group.accuracy.max() == group.accuracy].index[0])

            final_layer_results = results.iloc[final_layer_indices]
            df = final_layer_results.merge(networks, on='model')
            df.to_csv('all.csv', index=False)

            df.loc[df.training.str.startswith('SSL'), 'training'] = 'Self-Supervised'

            df.imagenet_accuracy /= 100.0

            counts = df.training.value_counts()
            for name, count in zip(counts.index, counts):
                print(name, count)
                df.loc[df.training == name, 'count'] = count

            df = df.sort_values(by='count', ascending=False)

            ax = sns.scatterplot(
                data=df,
                x=args.x_metric,
                y="accuracy",
                hue="training",
                style="training",
                s=MARKER_SIZE,
                alpha=0.6,
                legend="full",
                hue_order=PALETTE.keys(),
                style_order=PALETTE.keys(),
                palette=PALETTE,
            )

            ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
            #ax.set_xticks(np.arange(x_lim[0], x_lim[1], 5), fontsize=TICKS_LABELSIZE)
            ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
            ax.set_ylabel(args.ylabels[idx], fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)

            ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
            if idx > 0:
                ax.legend(title="", ncol=1, loc=legend_locs[idx], fancybox=True, fontsize=20)
            else:
                ax.legend([], [], frameon=False)


            regr = linear_model.LinearRegression()
            length = len(df)
            regr.fit(df.imagenet_accuracy.values.reshape((length, 1)),
                     df.accuracy.values.reshape((length, 1)))
            lims = np.array(x_lim)
            # now plot both limits against each other
            ax.plot(lims, regr.predict(np.array(lims).reshape(-1, 1)), "--", alpha=0.8, color="grey", zorder=0)
            ax.margins(x=0)

    """handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.97), fontsize=20)"""
    plt.savefig(args.output, bbox_inches='tight')
