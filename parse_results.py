from utils.analyses import parse_results_dir
from os.path import join
import pickle
from utils.analyses.training_mapping import Mapper
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--results-root', default='resources/results')
args = parser.parse_args()

DATASETS = ['multi-arrangement', 'free-arrangement/set1', 'free-arrangement/set2', 'things']


def filter_models(df):
    # SSL models only have penultimate layer
    ssl_and_logits = (df.source == 'ssl') & (df.module == 'logits')
    df = df[~ssl_and_logits]

    ssl_and_logits = (df.family == 'SSL') & (df.module == 'logits')
    df = df[~ssl_and_logits]

    # Remove inceptionv3 model
    df = df[df.model != 'inception_v3']

    # We only look at vit-same results
    df = df[df.source != 'vit_best']

    model_blacklist = []
    for model in df.model.values:
        if model.startswith('resnet_v1_50_tpu_random_init') or 'seed1' in model:
            model_blacklist.append(model)
    df = df[~df.model.isin(model_blacklist)]
    return df


for dataset in DATASETS:
    if dataset == 'things':
        with open(join(args.results_root, dataset, 'zero-shot', 'all_results.pkl'), 'rb') as f:
            zero_shot_results = pickle.load(f)
        zero_shot_results = zero_shot_results.drop(columns=['choices', 'entropies', 'probas'])
        mapper = Mapper(zero_shot_results)
        zero_shot_results['training'] = mapper.get_training_objectives()

    else:
        zero_shot_results = parse_results_dir(join(args.results_root, dataset, 'zero-shot'))

    zero_shot_results = filter_models(zero_shot_results)
    # Write results to csv
    zero_shot_results.to_csv(join(args.results_root, dataset, 'zero-shot.csv'))

    if dataset == 'things':
        with open(join(args.results_root, dataset, 'transform', 'best_probing_results_without_norm_no_ooo_choices.pkl'),
                  'rb') as f:
            transform_results = pickle.load(f)
        transform_results['training'] = Mapper(transform_results).get_training_objectives()
    else:
        transform_results = parse_results_dir(join(args.results_root, dataset, 'transform'))

    transform_results = filter_models(transform_results)
    # Write results to csv
    transform_results.to_csv(join(args.results_root, dataset, 'transform.csv'))

    if dataset == 'things':
        with open(join(args.results_root, dataset, 'zero-shot', 'all_resultsdot.pkl'), 'rb') as f:
            dot_results = pickle.load(f)
        dot_results['training'] = Mapper(dot_results).get_training_objectives()
        dot_results = dot_results.drop(columns=['choices', 'entropies', 'probas'])

        old_results = pd.read_csv('resources/results/things/zero-shot/dot.csv')
        old_results = old_results[~old_results.model.isin(dot_results.model.values)]
        dot_results['accuracy'] = dot_results['zero-shot']
        dot_results = dot_results.drop(columns=['zero-shot'])
        dot_results = pd.concat((old_results, dot_results))
        dot_results = filter_models(dot_results)

        dot_results = dot_results[~dot_results.model.isin(['clip-ViT', 'clip-RN', 'r50-rotnet',
                                                           'r50-barlowtwins', 'r50-swav', 'r50-mocov2', 'r50-vicreg',
                                                           'r50-simclr', 'r50-jigsaw'])]
        dot_results.to_csv(join(args.results_root, dataset, 'dot.csv'))


