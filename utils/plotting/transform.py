import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from utils.plotting import PALETTE, metric_to_ylabel

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
y_lim = [0.0, 1.0]


def zero_shot_vs_transform_plot(zero_shot, transform, y_metric, dataset):
    f = plt.figure(figsize=(14, 10), dpi=200)
    gs = f.add_gridspec(1, 1)

    sns.set_context("paper")

    if dataset == 'things':
        zero_shot['accuracy'] = zero_shot['zero-shot']
        transform['accuracy'] = transform['probing']

    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])
        zero_shot = zero_shot[zero_shot.module == 'penultimate']
        transform = transform[transform.module == 'penultimate']
        transformed_short = pd.DataFrame({
            'model': transform.model.values,
            'transformed': transform[y_metric].values
        })
        results = zero_shot.merge(transformed_short, on='model')

        ax = sns.scatterplot(
            data=results,
            x=y_metric,
            y='transformed',
            s=MARKER_SIZE,
            hue="training",
            style="training",
            alpha=0.6,
            legend="full",
            hue_order=PALETTE.keys(),
            style_order=PALETTE.keys(),
            palette=PALETTE,
        )

        y_lim = [0, 1.0]
        if dataset == 'things':
            y_lim = [0.3, 0.7]
            linestyle = 'dotted'
            ax.set_ylim(*y_lim)
            ax.axhline(y=0.673, color='magenta', linestyle=linestyle)
            ax.axhline(y=0.333, color='k', linestyle=linestyle)

        ax.set_ylim(*y_lim)

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]

        # now plot both limits against each other
        ax.plot(lims, lims, "--", alpha=0.8, color="grey", zorder=0)
        ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)

        label = metric_to_ylabel(y_metric)

        ax.set_ylabel(label + ' (+ transform)', fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.set_xlabel(label, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.legend(ncol=1, loc='lower right', fancybox=True, fontsize=20,
                  title_fontsize=20, prop={'size': 24},
                  markerscale=3)
        ax.margins(x=0)
        return f
