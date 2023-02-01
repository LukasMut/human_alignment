import argparse
import os
from os.path import join
import pandas as pd
from utils.plotting import overview_plot, make_detail_plot, \
    zero_shot_vs_transform_plot, clip_plot, logits_penultimate_plot, distance_fn_plot


def generate_plot(results, plot_type, y_metric, output_dir, dataset, export_format='.pdf', prefix=''):
    networks = pd.read_csv('networks.csv')
    if plot_type == 'overview':
        fig = overview_plot(results=results, network_metadata=networks, y_metric=y_metric,
                            dataset=dataset)
    elif plot_type in ['ssl', 'imagenet', 'loss', 'scaling']:
        fig = make_detail_plot(results=results, network_metadata=networks, y_metric=y_metric,
                               dataset=dataset, plot_type=plot_type)
    elif plot_type == 'clip':
        fig = clip_plot(results=results, network_metadata=networks, y_metric=y_metric,
                        x_metric='imagenet_accuracy', dataset=dataset)
    elif plot_type == 'logits_penultimate':
        fig = logits_penultimate_plot(results=results, network_metadata=networks, y_metric=y_metric,
                                      x_metric='imagenet_accuracy', dataset=dataset)
    else:
        raise ValueError('Unknown plot type.')
    fig.savefig(join(output_dir, prefix + plot_type + export_format), bbox_inches='tight')


PLOT_TYPES = ['overview', 'loss', 'imagenet', 'scaling', 'ssl', 'clip', 'logits_penultimate', 'distance_fn']
DATASETS = ['cifar100-coarse', 'multi-arrangement', 'free-arrangement/set1', 'free-arrangement/set2', 'things']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=DATASETS + ['all'], default='all')
    parser.add_argument('--type', choices=PLOT_TYPES + ['all'], default='distance_fn')
    parser.add_argument('--export-format', choices=['.png', '.pdf'], default='.pdf')
    parser.add_argument('--root', default='resources')
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    plot_types = PLOT_TYPES if args.type == 'all' else [args.type]

    for dataset in datasets:
        for plot_type in plot_types:
            input_dir = join(args.root, 'results', dataset)
            output_dir = join(args.root, 'plots', dataset)
            os.makedirs(output_dir, exist_ok=True)

            if dataset in ['things', 'cifar100-coarse']:
                y_metrics = ['accuracy']
                prefixes = ['accuracy_']
            else:
                y_metrics = ['spearman_rho_correlation', 'pearson_corr_correlation']
                prefixes = ['spearman_', 'pearson_']

            if plot_type == 'distance_fn':
                if dataset == 'things':
                    cosine_results = pd.read_csv(join(input_dir, 'zero-shot.csv'))
                    dot_results = pd.read_csv(join(input_dir, 'dot.csv'))
                    for module in ['penultimate', 'logits']:
                        fig = distance_fn_plot(cosine_results=cosine_results,
                                               dot_results=dot_results,
                                               module=module)
                        fig.savefig(join(output_dir, plot_type + '_' + module), bbox_inches='tight')

            else:
                for y_metric, prefix in zip(y_metrics, prefixes):
                    zero_shot_results = pd.read_csv(join(input_dir, 'zero-shot.csv'))
                    generate_plot(zero_shot_results, plot_type=plot_type,
                                  y_metric='zero-shot' if dataset == 'things' else y_metric,
                                  output_dir=output_dir, export_format=args.export_format,
                                  dataset=dataset,
                                  prefix=prefix)

                    if dataset not in ['cifar100-coarse']:
                        transform_results = pd.read_csv(join(input_dir, 'transform.csv'))
                        generate_plot(transform_results, plot_type=plot_type,
                                      y_metric='probing' if dataset == 'things' else y_metric,
                                      output_dir=output_dir, prefix=prefix + 'transform_',
                                      export_format=args.export_format, dataset=dataset)
                        fig = zero_shot_vs_transform_plot(zero_shot=zero_shot_results,
                                                          transform=transform_results, y_metric=y_metric,
                                                          dataset=dataset)
                        fig.savefig(join(output_dir, prefix + 'zshot_vs_transform' + args.export_format),
                                    bbox_inches='tight')
