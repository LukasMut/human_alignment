import argparse
import numpy as np
import pickle
import pandas as pd
from analyses.dimensions import TSNEReduction
import matplotlib.pyplot as plt
import os
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('--features-path',
                    default='resources/vice_embedding.npy')
parser.add_argument('--concepts-path', default='resources/things_concepts.tsv')
parser.add_argument('--model', default='vice')
parser.add_argument('--output-dir', default='resources/plots/things/embeddings')
args = parser.parse_args()

concepts = pd.read_csv(args.concepts_path, delimiter='\t')

if args.model == 'vice':
    features = np.load(args.features_path)
else:
    with open(args.features_path, 'rb') as f:
        features = pickle.load(f)
    features = features[args.model]

categories = ['animal', 'vehicle', 'clothing', 'plant',
              'food', 'furniture', 'container']
colors = ['red', 'green', 'orange', 'blue', 'brown', 'purple', 'pink']

c = np.zeros(features.shape[0])
for i, category in enumerate(categories):
    subset = concepts[concepts["Top-down Category (WordNet)"] == category]
    c[subset.index] = i + 1

reducer = TSNEReduction(features)
X = reducer.compute()
plt.scatter(*zip(*X[c == 0]), c='black', label='other', alpha=.1)
for i, category in enumerate(categories):
    plt.scatter(*zip(*X[c == i + 1]), c=colors[i], label=category)
plt.axis('off')
plt.legend(loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.13))
os.makedirs(args.output_dir, exist_ok=True)
plt.savefig(join(args.output_dir, args.model + '.png'))
plt.show()
