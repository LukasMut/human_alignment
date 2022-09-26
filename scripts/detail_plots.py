import matplotlib.pyplot as plt
import argparse
import pandas as pd
import pathlib
import os
import seaborn as sns

ssl_mapping = {
    'r50-barlowtwins': 'Barlow Twins',
    'r50-vicreg': 'VicReg',
    'r50-swav': 'SwAV',
    'r50-simclr': 'SimCLR',
    'r50-mocov2': 'MoCoV2',
    'r50-rotnet': 'RotNet',
    'r50-jigsaw': 'Jigsaw Puzzle',
    'resnet50': 'Supervised',
}

loss_mapping = {
    'resnet_v1_50_tpu_softmax_seed0': {
        'name': 'Softmax',
        'R2': 0.349
    },
    'resnet_v1_50_tpu_label_smoothing_seed0': {
        'name': 'Label smooth.',
        'R2': 0.420
    },
    'resnet_v1_50_tpu_sigmoid_seed0': {
        'name': 'Sigmoid',
        'R2': 0.427
    },
    'resnet_v1_50_tpu_extra_wd_seed0': {
        'name': 'Extra L2',
        'R2': 0.572
    },
    'resnet_v1_50_tpu_dropout_seed0': {
        'name': 'Dropout',
        'R2': 0.461
    },
    'resnet_v1_50_tpu_logit_penalty_seed0': {
        'name': 'Logit penalty',
        'R2': 0.601
    },
    'resnet_v1_50_tpu_nt_xent_weights_seed0': {
        'name': 'Cosine softmax',
        'R2': 0.641
    },
    'resnet_v1_50_tpu_squared_error_seed0': {
        'name': 'Squared error',
        'R2': 0.845
    },
    'resnet_v1_50_tpu_mixup_seed0': {
        'name': 'MixUp',
    },
    'resnet_v1_50_tpu_nt_xent_seed0': {
        'name': 'Logit norm'
    },
    'resnet_v1_50_tpu_autoaugment_seed0': {
        'name': 'AutoAugment'
    }
}

LEGEND_FONT_SIZE = 24
X_AXIS_FONT_SIZE = 35
Y_AXIS_FONT_SIZE = 35
TICKS_LABELSIZE = 25
MARKER_SIZE = 400
AXIS_LABELPAD = 25
y_lim = [0.3, 0.6]
x_lim = [0.71, 0.82]
xlabel = "ImageNet accuracy"
global ylabel
DEFAULT_SCATTER_PARAMS = dict(s=MARKER_SIZE,
                              alpha=0.9,
                              legend="full")


def configure_axis(ax, limits=True, show_ylabel=True):
    if limits:
        ax.set_ylim(y_lim)
        ax.set_xlim(x_lim)
    ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
    ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
    else:
        ax.set_ylabel("", fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
    ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)


def plot_imagenet_models(df):
    df = df[df.source == 'imagenet']
    print(len(df))
    ax = sns.scatterplot(
        data=df,
        x=args.x_metric,
        y="accuracy",
        hue="family",
        style="family",
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax, show_ylabel=False)
    ax.legend(title="",
              ncol=1,
              loc="lower right",
              fancybox=True,
              fontsize=LEGEND_FONT_SIZE)


def plot_loss_models(df):
    df = df[df.source == 'loss']
    df = df.assign(model=df.model.map(lambda x: loss_mapping[x]['name']))

    ax = sns.scatterplot(
        data=df,
        x=args.x_metric,
        y="accuracy",
        hue="model",
        style="model",
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax)
    ax.legend(title="", ncol=1,
              loc="upper left",
              fancybox=True,
              fontsize=LEGEND_FONT_SIZE)


def plot_ssl_models(df):
    df = df[df.training == 'Self-Supervised']
    df = df.assign(model=df.model.map(lambda x: ssl_mapping[x]))

    ax = sns.scatterplot(
        data=df,
        x=args.x_metric,
        y="accuracy",
        hue="model",
        style="model",
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax, limits=False)
    ax.legend(title="", ncol=1,
              loc="upper left",
              fancybox=True,
              fontsize=LEGEND_FONT_SIZE)


def plot_arch_models(df):
    selectors = [('VGG', df.family.isin(['VGG'])),
                 ('EfficientNet', df.family.isin(['EfficientNet'])),
                 ('ResNet', ((df.family == 'ResNet') & (df.source == 'torchvision'))),
                 ('VIT (ImageNet-1K)', (df.training == 'Supervised (ImageNet-1K)') & (df.source == 'vit_same')),
                 ('VIT (ImageNet-21K)', (df.training == 'Supervised (ImageNet-21K)') & (df.source == 'vit_same'))
                 ]

    df['label'] = 'None'
    all_selectors = None
    for name, s in selectors:
        if all_selectors is None:
            all_selectors = s
        else:
            all_selectors |= s
        df.loc[s, 'label'] = name

    df = df[all_selectors]

    ax = sns.scatterplot(
        data=df,
        x='param_count',
        y="accuracy",
        hue="label",
        style="label",
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax, limits=False, show_ylabel=False)
    ax.set_xlabel('#Parameters (Million)', fontsize=X_AXIS_FONT_SIZE)
    ax.legend(title="", ncol=1, loc="upper right", fancybox=True, fontsize=LEGEND_FONT_SIZE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='resources/final_results/things/results.csv')
    parser.add_argument('--network_data_path', default=None)
    parser.add_argument('--x_metric', default='imagenet_accuracy', choices=['imagenet_accuracy', 'param_count'])
    parser.add_argument('--output', default='resources/final_results/plots/things')
    parser.add_argument('--ylabel', default='Zero-shot odd-one-out accuracy')
    args = parser.parse_args()

    ylabel = args.ylabel

    if args.network_data_path is None:
        network_data_path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'networks.csv')
    else:
        network_data_path = args.network_data_path
    networks = pd.read_csv(network_data_path)

    results_path = args.path
    results = pd.read_csv(args.path)
    print(len(results[results.source == 'imagenet']))
    final_layer_indices = []
    k = 0
    for name, group in results.groupby('model'):
        if 'imagenet' in group.source.values.tolist():
            print(name)
            k += 1
        if len(group.index) > 2:
            sources = group.source.values.tolist()
            assert 'vit_best' in sources and 'vit_same' in sources
            group = group[group.source == 'vit_same']
        final_layer_indices.append(group[group.accuracy.max() == group.accuracy].index[0])

    print(k)

    final_layer_results = results.iloc[final_layer_indices]
    df = final_layer_results.merge(networks, on='model')

    df.loc[df.training.str.startswith('SSL'), 'training'] = 'Self-Supervised'
    df.imagenet_accuracy /= 100.0

    f = plt.figure(figsize=(28, 10), dpi=200)
    gs = f.add_gridspec(1, 2)
    sns.set_context("talk")
    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])
        plot_loss_models(df)
        f.add_subplot(gs[0, 1])
        plot_imagenet_models(df)
    f.tight_layout()
    plt.savefig(os.path.join(args.output, 'loss-imagenet.pdf'))
    plt.clf()

    f = plt.figure(figsize=(28, 10), dpi=200)
    gs = f.add_gridspec(1, 2)
    sns.set_context("talk")
    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])
        plot_ssl_models(df)
        f.add_subplot(gs[0, 1])
        plot_arch_models(df)
    f.tight_layout()
    plt.savefig(os.path.join(args.output, 'ssl-scaling.pdf'))
