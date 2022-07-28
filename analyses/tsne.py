import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class DimensionReduction:

    def __init__(self, embeddings: np.ndarray):
        self.embeddings = embeddings

    def compute(self):
        pass


class TSNEReduction(DimensionReduction):

    def compute(self):
        tsne = TSNE(n_components=2, n_iter=2000, learning_rate='auto', init='random', metric='cosine', random_state=0)
        X = tsne.fit_transform(self.embeddings)
        return X