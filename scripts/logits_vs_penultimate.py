import sys

sys.path.append('.')

import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pathlib
import os
import seaborn as sns
import numpy as np
from utils.analyses.helpers import map_objective_function

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
things_y_lim = [0.35, 0.55]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network_data_path', default=None)
    parser.add_argument('--output', default='resources/final_results/plots/logits_penultimate.pdf')
    parser.add_argument('--legend-loc', default="lower left")
    parser.add_argument('--path', default='resources/final_results/things/results.csv')
    args = parser.parse_args()

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    legend_locs = ['lower left', 'upper left']

    f = plt.figure(figsize=(14, 10), dpi=200)
    gs = f.add_gridspec(1, 1)

    sns.set_context("paper")

    with sns.axes_style("ticks"):

        f.add_subplot(gs[0, 0])
        results = pd.read_csv(args.path)

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

                logits = group[group.module == 'logits'].accuracy.tolist()[0]
                penultimate = group[group.module == 'penultimate'].accuracy.tolist()[0]
                objective = map_objective_function(name, source, family)

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
        ax.set_xticks(np.arange(things_y_lim[0], things_y_lim[1], 0.05), fontsize=TICKS_LABELSIZE)
        ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.set_ylabel('Logits', fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.set_xlabel('Penultimate', fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.legend(title="Objective", ncol=1, loc='upper left', fancybox=True, fontsize=20, title_fontsize=20)
        ax.margins(x=0)

    plt.savefig(args.output, bbox_inches='tight')
