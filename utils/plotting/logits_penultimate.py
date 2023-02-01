import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils.analyses.helpers import map_objective_function
import pandas as pd
import matplotlib.ticker as plticker

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
things_y_lim = [0.0, 1.0]


def logits_penultimate_plot(results, network_metadata, y_metric, x_metric, dataset):
    sns.set_context("paper")
    f = plt.figure(figsize=(14, 10), dpi=200)
    gs = f.add_gridspec(1, 1)

    with sns.axes_style("ticks"):

        f.add_subplot(gs[0, 0])

        models = []
        for name, group in results.groupby('model'):
            if len(group.index) == 2:
                source = group.source.values.tolist()[0]
                family = group.family.values.tolist()[0]

                logits = group[group.module == 'logits'][y_metric].tolist()[0]
                penultimate = group[group.module == 'penultimate'][y_metric].tolist()[0]
                objective = map_objective_function(name, source, family)

                models.append({
                    'model': name,
                    'logits': logits,
                    'penultimate': penultimate,
                    'objective': objective
                })
            else:
                print(group)
                # raise RuntimeError()

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

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # now plot both limits against each other
        ax.plot(lims, lims, "--", alpha=0.8, color="grey", zorder=0)
        ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.set_ylabel('Logits', fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.set_xlabel('Penultimate', fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.legend(title="Objective", ncol=1, loc='upper left', fancybox=True, fontsize=20, title_fontsize=20)
        ax.margins(x=0)

        step_size = 0.1
        if dataset == 'things':
            step_size = 0.05
        loc = plticker.MultipleLocator(base=step_size)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)

    f.tight_layout()
    return f
