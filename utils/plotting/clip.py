import matplotlib.pyplot as plt
import seaborn as sns
from utils.plotting import metric_to_ylabel
from sklearn import linear_model
import numpy as np

NAME_MAPPING = {
    'clip_ViT-B/32': 'ViT-B/32',
    'clip_ViT-B/16': 'ViT-B/16',
    'clip_ViT-L/14': 'ViT-L/14',
    'clip_RN50': 'ResNet 50',
    'clip_RN101': 'ResNet 101',
    'clip_RN50x4': 'ResNet 50x4',
    'clip_RN50x16': 'ResNet 50x16',
    'clip_RN50x64': 'ResNet 50x64'
}


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


def clip_plot(results, network_metadata, y_metric, x_metric, dataset):
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
        df.loc[:, 'model'] = list(map(lambda x: NAME_MAPPING[x], df.model.values.tolist()))

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

        y_lim = [0, 1.0]
        if dataset == 'things':
            y_lim = [0.3, 0.7]
            linestyle = 'dotted'
            ax.set_ylim(*y_lim)
            ax.axhline(y=0.673, color='magenta', linestyle=linestyle)
            ax.axhline(y=0.333, color='k', linestyle=linestyle)

        ax.set_ylim(*y_lim)

        ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
        ax.set_ylabel(metric_to_ylabel(y_metric), fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)

        ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
        ax.legend(title="", ncol=1, loc='lower left', fancybox=True, fontsize=17)

        regr = linear_model.LinearRegression()
        length = len(df)
        regr.fit(df.imagenet_accuracy.values.reshape((length, 1)), df[y_metric].values.reshape((length, 1)))
        lims = np.array(x_lim)
        # now plot both limits against each other
        ax.plot(lims, regr.predict(np.array(lims).reshape(-1, 1)), "--", alpha=0.8, color="grey", zorder=0)
        ax.margins(x=0)
        plt.title('Image/text models', fontsize=30, pad=20)
    f.tight_layout()
    return f
