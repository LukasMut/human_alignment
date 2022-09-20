import pickle
import matplotlib.pyplot as plt
import argparse
import os

import pandas as pd
import seaborn as sns

mapping = {
    'r50-barlowtwins': "ResNet-50-BarlowTwins",
    'vgg19': 'VGG-19',
    'alexnet': 'AlexNet',
    'resnet50': 'ResNet-50',
    'resnet152': 'ResNet-152',
    'vit_b_16': 'VIT-B/16',
    'vit_l_16': 'VIT-L/16'
}


def load_dataframe(path: str):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


LEGEND_FONT_SIZE = 24
X_AXIS_FONT_SIZE = 35
Y_AXIS_FONT_SIZE = 35
TICKS_LABELSIZE = 25
MARKER_SIZE = 400
AXIS_LABELPAD = 25
xlabel = "Layer Depth"
ylabel = 'Zero-shot odd-one-out accuracy'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--path', default=None)
    group.add_argument('--csv', default='resources/layers.csv')
    parser.add_argument('--output', default='resources/final_results/plots/layer_plot.pdf')
    args = parser.parse_args()

    if args.path is not None:
        results = []
        for subdir in os.listdir(args.path):
            df = load_dataframe(os.path.join(args.path, subdir, 'results.pkl'))
            df['depth'] = df.index.values
            results.append(df)
        df = pd.concat(results)[['model', 'layer', 'accuracy', 'depth']]
        df = df.reset_index()
    else:
        df = pd.read_csv(args.csv)

    df = df[~df.model.isin(['vit_b_16'])]
    df = df.assign(model=df.model.map(lambda x: mapping[x]))

    sns.set_context("paper")
    f = plt.figure(figsize=(14, 10), dpi=200)
    gs = f.add_gridspec(1, 2)
    sns.set_context("talk")

    with sns.axes_style("ticks"):
        ax = sns.lineplot(
            data=df,
            x='depth',
            y="accuracy",
            hue="model",
            style="model",
            markers=True,
            alpha=0.9,
            legend="full",
            markersize=15.0,
            linewidth=5.0
        )
    ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
    ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)

    ax.set_ylabel(ylabel, fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
    ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)

    ax.legend(title="", ncol=2,
              loc="upper left",
              fancybox=True,
              fontsize=LEGEND_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(args.output)
