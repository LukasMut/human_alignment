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
things_y_lim = [0.35, 0.55]

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
    parser.add_argument('--only_two', action='store_true', default=True)
    args = parser.parse_args()

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    legend_locs = ['upper left', 'upper left']

    sns.set_context("paper")
    if len(args.paths) == 2:
        if args.only_two:
            f = plt.figure(figsize=(28, 10), dpi=200)
            gs = f.add_gridspec(1, 2)
        else:
            f = plt.figure(figsize=(42, 10), dpi=200)
            gs = f.add_gridspec(1, 3)
    else:
        f = plt.figure(figsize=(14, 10), dpi=200)
        gs = f.add_gridspec(1, 1)

    sns.set_context("talk")
    with sns.axes_style("ticks"):
        for idx, path in enumerate(args.paths):
            f.add_subplot(gs[0, idx if args.only_two else idx + 1])
            results = pd.read_csv(path)
            final_layer_indices = []
            for name, group in results.groupby('model'):
                if 'seed1' in name:
                    continue
                if len(group.index) > 2:
                    sources = group.source.values.tolist()
                    assert 'vit_best' in sources and 'vit_same' in sources
                    group = group[group.source == 'vit_same']
                final_layer_indices.append(group[group.accuracy.max() == group.accuracy].index[0])

            final_layer_results = results.iloc[final_layer_indices]
            df = final_layer_results.merge(networks, on='model')
            df.to_csv(f'all_{idx}.csv', index=False)

            df.loc[df.training.str.startswith('SSL'), 'training'] = 'Self-Supervised'

            df.imagenet_accuracy /= 100.0

            counts = df.training.value_counts()
            for name, count in zip(counts.index, counts):
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
            if not args.only_two:
                if idx == 0:
                    ax.set_yticks(np.arange(things_y_lim[0], things_y_lim[1], 0.05), fontsize=TICKS_LABELSIZE)
                    ax.set_ylim([things_y_lim[0] - 0.02, things_y_lim[1]])

            ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
            ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
            ax.set_ylabel(args.ylabels[idx], fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)

            ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
            if idx > 0 or len(args.paths) == 1:
                ax.legend(title="", ncol=1, loc=legend_locs[idx], fancybox=True, fontsize=17)
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

        if not args.only_two:
            f.add_subplot(gs[0, 0])
            results = pd.read_csv(args.paths[0])

            models = []
            for name, group in results.groupby('model'):
                if 'seed1' in name or name.startswith('resnet_v1_50_tpu_random_init'):
                    continue
                if len(group.index) > 2:
                    sources = group.source.values.tolist()
                    assert 'vit_best' in sources and 'vit_same' in sources
                    group = group[group.source == 'vit_same']
                if len(group.index) == 2:
                    source = group.source.values.tolist()[0]
                    family = group.family.values.tolist()[0]
                    objective = 'Supervised (softmax)'
                    if source == 'vit_same' or name in ['resnet_v1_50_tpu_sigmoid_seed0']:
                        objective = 'Supervised (sigmoid)'
                    elif name in ['resnet_v1_50_tpu_nt_xent_weights_seed0',
                                  'resnet_v1_50_tpu_nt_xent_seed0',
                                  'resnet_v1_50_tpu_label_smoothing_seed0',
                                  'resnet_v1_50_tpu_logit_penalty_seed0',
                                  'resnet_v1_50_tpu_mixup_seed0',
                                  'resnet_v1_50_tpu_extra_wd_seed0']:
                        objective = 'Supervised (softmax+)'
                    elif family.startswith('SSL'):
                        objective = 'Self-sup.'
                    elif source == 'loss' and name == 'resnet_v1_50_tpu_squared_error_seed0':
                        objective = 'Supervised (squared error)'
                    elif family in ['Align', 'Basic', 'CLIP']:
                        objective = 'Image/Text (contrastive)'

                    logits = group[group.module == 'logits'].accuracy.tolist()[0]
                    penultimate = group[group.module == 'penultimate'].accuracy.tolist()[0]

                    if penultimate > logits:
                        print(objective, name)

                    models.append({
                        'model': name,
                        'logits': logits,
                        'penultimate': penultimate,
                        'objective': objective
                    })
            module_df = pd.DataFrame(models)
            ax = sns.scatterplot(
                data=module_df,
                x="penultimate",
                y="logits",
                s=MARKER_SIZE,
                alpha=0.6,
                hue="objective",
                style="objective",
                legend="full"
            )
            ax.set_ylim([things_y_lim[0] - 0.02, things_y_lim[1]])
            ax.set_xlim([things_y_lim[0] - 0.02, things_y_lim[1]])

            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]

            # now plot both limits against each other
            ax.plot(lims, lims, "--", alpha=0.8, color="grey", zorder=0)
            ax.set_yticks(np.arange(things_y_lim[0], things_y_lim[1], 0.05), fontsize=TICKS_LABELSIZE)
            ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
            ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
            ax.set_ylabel('Logits', fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
            ax.set_xlabel('Penultimate', fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
            ax.legend(title="Objective", ncol=1, loc='upper left', fancybox=True, fontsize=17)

            plt.subplots_adjust(wspace=0.25)

    """handles, labels = ax.get_legend_handles_labels()
    f.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 0.97), fontsize=20)"""
    plt.savefig(args.output, bbox_inches='tight')
