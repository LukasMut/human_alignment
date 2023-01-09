import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import metric_to_ylabel
from sklearn import linear_model
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
things_y_lim = [0.0, 1.0]


def clip_plot(results, network_metadata, y_metric, x_metric):
    sns.set_context("paper")
    f = plt.figure(figsize=(14, 10), dpi=200)
    gs = f.add_gridspec(1, 1)

    sns.set_context("talk")
    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])
        results = results[results.family == 'CLIP']
        df = results.merge(network_metadata, on='model')
        df.imagenet_accuracy /= 100.0
        counts = df.training.value_counts()
        for name, count in zip(counts.index, counts):
            df.loc[df.training == name, 'count'] = count

        df = df.sort_values(by='count', ascending=False)

        ax = sns.scatterplot(
            data=df,
            x=x_metric,
            y=y_metric,
            hue="model",
            style="model",
            s=MARKER_SIZE,
            alpha=0.6,
            legend="full",
        )
        linestyle = 'dotted'

        ax.set_ylim([things_y_lim[0], things_y_lim[1]])

        ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.set_ylabel(metric_to_ylabel(y_metric), fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)

        ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.legend(title="", ncol=1, loc='upper left', fancybox=True, fontsize=17)

        regr = linear_model.LinearRegression()
        length = len(df)
        regr.fit(df.imagenet_accuracy.values.reshape((length, 1)), df[y_metric].values.reshape((length, 1)))
        lims = np.array(x_lim)
        # now plot both limits against each other
        ax.plot(lims, regr.predict(np.array(lims).reshape(-1, 1)), "--", alpha=0.8, color="grey", zorder=0)
        ax.margins(x=0)
    return f
