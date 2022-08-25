from dataclasses import dataclass

import numpy as np
from sklearn.manifold import TSNE

Array = np.ndarray


@dataclass
class DimensionReduction:
    embeddings: Array

    def compute(self):
        pass


class TSNEReduction(DimensionReduction):
    def compute(self):
        tsne = TSNE(
            n_components=2,
            n_iter=2000,
            learning_rate="auto",
            init="random",
            metric="cosine",
            random_state=0,
        )
        X = tsne.fit_transform(self.embeddings)
        return X
