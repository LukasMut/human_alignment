import argparse
import numpy as np
import pandas as pd
from analyses.tsne import TSNEReduction
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--features-path', default='/Users/jdippel/Development/research/cognitive-embeddings/resources/results/logits/resnets/vgg11/classifier.6/features.npy')
parser.add_argument('--concepts-path', default='resources/things_concepts.tsv')
args = parser.parse_args()

concepts = pd.read_csv(args.concepts_path, delimiter='\t')
features = np.load(args.features_path)

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
#plt.title('CLIP VIT')
plt.show()


