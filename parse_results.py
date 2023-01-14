from utils.analyses import parse_results_dir
from os.path import join
import pickle
from utils.analyses.training_mapping import Mapper
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results-root', default='resources/results')
args = parser.parse_args()

DATASETS = ['multi-arrangement', 'free-arrangement/set1', 'free-arrangement/set2', 'things']


def filter_models(df):
    # SSL models only have penultimate layer
    ssl_and_logits = (df.source == 'ssl') & (df.module == 'logits')
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
