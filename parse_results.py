from utils.analyses import parse_results_dir
from os.path import join
import pickle
from utils.analyses.training_mapping import Mapper

DATASETS = ['multi-arrangement', 'free-arrangement/set1', 'free-arrangement/set2', 'things']
RESULTS_DIR = 'resources/results'

for dataset in DATASETS:
    if dataset == 'things':
        with open(join(RESULTS_DIR, dataset, 'zero-shot', 'all_results.pkl'), 'rb') as f:
            zero_shot_results = pickle.load(f)
        zero_shot_results = zero_shot_results.drop(columns=['choices', 'entropies', 'probas'])
        # SSL models only have penultimate layer
        ssl_and_logits = (zero_shot_results.source == 'ssl') & (zero_shot_results.module == 'logits')
        zero_shot_results = zero_shot_results[~ssl_and_logits]
        # Remove inceptionv3 model
        zero_shot_results = zero_shot_results[zero_shot_results.model != 'inception_v3']
        mapper = Mapper(zero_shot_results)
        zero_shot_results['training'] = mapper.get_training_objectives()

    else:
        zero_shot_results = parse_results_dir(join(RESULTS_DIR, dataset, 'zero-shot'))

    # We only look at vit-same results
    zero_shot_results = zero_shot_results[zero_shot_results.source != 'vit_best']

    # Write results to csv
    zero_shot_results.to_csv(join(RESULTS_DIR, dataset, 'zero-shot.csv'))

    if dataset != 'things':
        transform_results = parse_results_dir(join(RESULTS_DIR, dataset, 'transform'))

        # Write results to csv
        transform_results.to_csv(join(RESULTS_DIR, dataset, 'transform.csv'))
