import copy
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import numpy as np
from utils.plotting import metric_to_ylabel, reduce_best_final_layer


def map_objective_function(model_name, source, family):
    objective = 'Softmax'
    if source == 'vit_same' or model_name in ['resnet_v1_50_tpu_sigmoid_seed0']:
        objective = 'Sigmoid'
    elif model_name in ['resnet_v1_50_tpu_nt_xent_weights_seed0',
                        'resnet_v1_50_tpu_nt_xent_seed0',
                        'resnet_v1_50_tpu_label_smoothing_seed0',
                        'resnet_v1_50_tpu_logit_penalty_seed0',
                        'resnet_v1_50_tpu_mixup_seed0',
                        'resnet_v1_50_tpu_extra_wd_seed0']:
        objective = 'Softmax+'
    elif family.startswith('SSL'):
        objective = 'Self-sup.'
    elif source == 'loss' and model_name == 'resnet_v1_50_tpu_squared_error_seed0':
        objective = 'Squared error'
    elif family in ['Align', 'Basic', 'CLIP']:
        objective = 'Image/Text (contrastive)'
    return objective


ssl_mapping = {
    'barlowtwins-rn50': 'Barlow Twins',
    'vicreg-rn50': 'VicReg',
    'swav-rn50': 'SwAV',
    'simclr-rn50': 'SimCLR',
    'mocov2-rn50': 'MoCoV2',
    'rotnet-rn50': 'RotNet',
    'jigsaw-rn50': 'Jigsaw Puzzle',
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
xlabel = "ImageNet accuracy"
DEFAULT_SCATTER_PARAMS = dict(s=MARKER_SIZE,
                              legend="full")
ALPHA = 0.5

y_lim = [0.0, 1.0]

x_lim = [0.71, 0.82]


def configure_axis(ax, y_metric, limit_x=True, limit_y=True, show_ylabel=True):
    ylabel = metric_to_ylabel(y_metric)
    if limit_y:
        ax.set_ylim(y_lim)
    if limit_x:
        ax.set_xlim(x_lim)
    ax.xaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
    ax.yaxis.set_tick_params(labelsize=TICKS_LABELSIZE)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
    else:
        ax.set_ylabel("", fontsize=Y_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)
    ax.set_xlabel(xlabel, fontsize=X_AXIS_FONT_SIZE, labelpad=AXIS_LABELPAD)


def plot_imagenet_models(df, x_metric, y_metric):
    df = df[df.source == 'imagenet']
    ax = sns.scatterplot(
        data=df,
        x=x_metric,
        y=y_metric,
        hue="family",
        style="family",
        alpha=ALPHA,
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax, show_ylabel=False, y_metric=y_metric)
    ax.legend(title="",
              ncol=1,
              loc='upper left',
              fancybox=True,
              fontsize=LEGEND_FONT_SIZE)
    plt.title('Varying architecture', fontsize=30, pad=20)


def augmentation_strategy(x):
    if x == 'resnet_v1_50_tpu_mixup_seed0':
        return 'MixUp'
    elif x == 'resnet_v1_50_tpu_autoaugment_seed0':
        return 'AutoAugment'
    else:
        return 'Default'


def objective(x):
    if x['model'] in ['resnet_v1_50_tpu_softmax_seed0', 'resnet_v1_50_tpu_autoaugment_seed0']:
        return 'Softmax'
    return loss_mapping[x['model']]['name']


def plot_loss_models(df, x_metric, y_metric):
    df = copy.deepcopy(df[df.source == 'loss'])
    df.loc[:, 'Augmentation'] = list(map(lambda x: augmentation_strategy(x), deepcopy(df.model.values)))
    df.loc[:, 'Objective'] = df.apply(lambda x: objective(x), axis=1)
    df = df.assign(model=df.model.map(lambda x: loss_mapping[x]['name']))

    ax = sns.scatterplot(
        data=df,
        x=x_metric,
        y=y_metric,
        hue="Objective",
        style="Augmentation",
        s=MARKER_SIZE,
        alpha=ALPHA,
    )
    configure_axis(ax, y_metric=y_metric)
    ax.plot(np.zeros(1), np.zeros([1, 6]), color='w', alpha=0, label=' ')
    ax.legend(title="", ncol=2,
              loc='upper left',
              fancybox=True,
              fontsize=17)
    plt.title('Varying objective', fontsize=30, pad=20)


def plot_ssl_models(df, x_metric, y_metric):
    df = df[df.training == 'Self-Supervised']
    df = df.assign(model=df.model.map(lambda x: ssl_mapping[x]))

    ax = sns.scatterplot(
        data=df,
        x=x_metric,
        y=y_metric,
        hue="model",
        style="model",
        alpha=ALPHA,
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax, limit_x=False, y_metric=y_metric)
    ax.legend(title="", ncol=2,
              fancybox=True,
              loc='upper left',
              fontsize=LEGEND_FONT_SIZE)
    plt.title('Self-sup. models', fontsize=30, pad=20)


def plot_arch_models(df, y_metric):
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
        y=y_metric,
        hue="label",
        alpha=ALPHA,
        style="label",
        **DEFAULT_SCATTER_PARAMS
    )
    configure_axis(ax, limit_x=False, show_ylabel=False, y_metric=y_metric)
    ax.set_xlabel('#Parameters (Million)', fontsize=X_AXIS_FONT_SIZE)
    ax.legend(title="", ncol=1,
              loc='upper right',
              fancybox=True, fontsize=LEGEND_FONT_SIZE)
    plt.title('Model size', fontsize=30, pad=20)


def rescale_accuracy(x):
    chance_level = 1 / 3.
    max_acc = 0.673
    return (x - chance_level) / (max_acc - chance_level)


def loss_imagenet_plot(results, network_metadata, y_metric, x_metric='imagenet_accuracy', legend_loc='upper left'):
    final_layer_results = reduce_best_final_layer(results, metric=y_metric)
    df = final_layer_results.merge(network_metadata, on='model')
    df.loc[df.training.str.startswith('SSL'), 'training'] = 'Self-Supervised'
    df.imagenet_accuracy /= 100.0

    f = plt.figure(figsize=(28, 10), dpi=200)
    gs = f.add_gridspec(1, 2)
    sns.set_context("talk")
    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])
        plot_loss_models(df, x_metric=x_metric, y_metric=y_metric)
        f.add_subplot(gs[0, 1])
        plot_imagenet_models(df, x_metric=x_metric, y_metric=y_metric)
    f.tight_layout()
    return f


def ssl_scaling_plot(results, network_metadata, y_metric, x_metric='imagenet_accuracy', legend_loc='upper left'):
    final_layer_results = reduce_best_final_layer(results, metric=y_metric)
    df = final_layer_results.merge(network_metadata, on='model')
    df.loc[df.training.str.startswith('SSL'), 'training'] = 'Self-Supervised'
    df.imagenet_accuracy /= 100.0

    f = plt.figure(figsize=(28, 10), dpi=200)
    gs = f.add_gridspec(1, 2)
    sns.set_context("talk")
    with sns.axes_style("ticks"):
        f.add_subplot(gs[0, 0])
        plot_ssl_models(df, x_metric=x_metric, y_metric=y_metric)
        f.add_subplot(gs[0, 1])
        plot_arch_models(df, y_metric=y_metric)
    f.tight_layout()
    return f
