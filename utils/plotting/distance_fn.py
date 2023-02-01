import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.ticker as plticker
from utils.plotting import PALETTE

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


def distance_fn_plot(cosine_results, dot_results, module='penultimate'):
    sns.set_context("paper")
    f = plt.figure(figsize=(14, 10), dpi=200)
    gs = f.add_gridspec(1, 1)

    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])

        dot_results['dot'] = dot_results['accuracy']
        cosine_results['cosine'] = cosine_results['zero-shot']
        dot_results = dot_results[['model', 'module', 'dot']]
        results = cosine_results.merge(dot_results, on=['model', 'module'])
        results = results[results.module == module]
        results.to_csv('compare_' + module + '.csv')

        ax = sns.scatterplot(
            data=results,
            x='dot',
            y="cosine",
            hue="training",
            style="training",
            s=MARKER_SIZE,
            alpha=0.6,
            legend="full",
            hue_order=PALETTE.keys(),
            style_order=PALETTE.keys(),
            palette=PALETTE,
        )

        loc = plticker.MultipleLocator(base=0.05)  # this locator puts ticks at regular intervals
        ax.xaxis.set_major_locator(loc)

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
        ax.set_ylabel('Cosine', fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.set_xlabel('Dot Product', fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.margins(x=0)
    f.tight_layout()
    return f
