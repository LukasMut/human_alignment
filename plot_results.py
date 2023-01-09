import argparse
import os
from os.path import join
import pandas as pd
from utils.plotting import overview_plot, loss_imagenet_plot, ssl_scaling_plot


def generate_plot(results, plot_type, y_metric, output_dir, export_format='.pdf', prefix=''):
    networks = pd.read_csv('networks_ma.csv')
    if plot_type == 'overview':
        fig = overview_plot(results=results, network_metadata=networks, y_metric=y_metric)
        fig.savefig(join(output_dir, prefix + 'overview' + export_format), bbox_inches='tight')
    elif plot_type == 'loss-imagenet':
        fig = loss_imagenet_plot(results=results, network_metadata=networks, y_metric=y_metric)
        fig.savefig(join(output_dir, prefix + 'loss-imagenet' + export_format), bbox_inches='tight')
        pass
    elif plot_type == 'ssl-scaling':
        fig = ssl_scaling_plot(results=results, network_metadata=networks, y_metric=y_metric)
        fig.savefig(join(output_dir, prefix + 'ssl-scaling' + export_format), bbox_inches='tight')
    else:
        raise ValueError('Unknown plot type.')


PLOT_TYPES = ['overview', 'loss-imagenet', 'ssl-scaling']
RESOURCES_FOLDER = 'resources'
DATASETS = ['multi-arrangement', 'free-arrangement/set1', 'free-arrangement/set2', 'things']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='free-arrangement/set1')
    parser.add_argument('--type', choices=PLOT_TYPES + ['all'], default='all')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    plot_types = PLOT_TYPES if args.type == 'all' else [args.type]

    for dataset in datasets:
        print(dataset)
        for plot_type in plot_types:
            input_dir = join(RESOURCES_FOLDER, 'results', dataset)
            output_dir = join(RESOURCES_FOLDER, 'plots', dataset)
            os.makedirs(output_dir, exist_ok=True)

            if dataset in ['things']:
                y_metric = 'zero-shot'
            else:
                y_metric = 'spearman_rho_correlation'

            zero_shot_results = pd.read_csv(join(input_dir, 'zero-shot.csv'))
            generate_plot(zero_shot_results, plot_type=plot_type, y_metric=y_metric,
                          output_dir=output_dir)

            if dataset != 'things':
                transform_results = pd.read_csv(join(input_dir, 'transform.csv'))
                generate_plot(transform_results, plot_type=plot_type, y_metric=y_metric,
                              output_dir=output_dir, prefix='transform_')
