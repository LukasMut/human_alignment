from typing import Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

Tensor = torch.Tensor


class Linear(pl.LightningModule):
    def __init__(
        self,
        features: Tensor,
        transform_dim: int,
        optim: str,
        lr: float,
        num_samples: int,
        model: str,
    ):
        super().__init__()
        self.features = torch.nn.Parameter(
            torch.from_numpy(features),
            requires_grad=False,
        )
        self.feature_dim = self.features.shape[1]

        if model.lower().startswith("vgg"):
            std = 1e-3
        else:
            std = 1e-2

        self.transform = torch.nn.Parameter(
            data=torch.normal(
                mean=torch.zeros(self.feature_dim, transform_dim),
                std=torch.ones(self.feature_dim, transform_dim) * std,
            ),
            requires_grad=True,
        )
        self.optim = optim
        self.lr = lr
        self.num_samples = num_samples
        self.lmbda = 1e-3  # (1 / self.num_samples)

    def forward(self, one_hots: Tensor) -> Tensor:
        embedding = self.features @ self.transform
        embeddings = one_hots @ embedding
        return embeddings

    @staticmethod
    def compute_similarities(
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply the similarity function (modeled as a dot product) to each pair in the triplet."""
        sim_i = torch.sum(anchor * positive, dim=1)
        sim_j = torch.sum(anchor * negative, dim=1)
        sim_k = torch.sum(positive * negative, dim=1)
        return (sim_i, sim_j, sim_k)

    @staticmethod
    def sumexp(sims: Tuple[Tensor]) -> Tensor:
        return torch.sum(torch.exp(torch.stack(sims)), dim=0)

    def softmax(self, sims: Tuple[Tensor]) -> Tensor:
        return torch.exp(sims[0]) / self.sumexp(sims)

    def logsumexp(self, sims: Tuple[Tensor]) -> Tensor:
        return torch.log(self.sumexp(sims))

    def log_softmax(self, sims: Tuple[Tensor]) -> Tensor:
        return sims[0] - self.logsumexp(sims)

    def cross_entropy_loss(self, sims: Tuple[Tensor]) -> Tensor:
        return torch.mean(-self.log_softmax(sims))

    @staticmethod
    def break_ties(probas: Tensor) -> Tensor:
        return torch.tensor(
            [
                -1 if torch.unique(pmf).shape[0] != pmf.shape[0] else torch.argmax(pmf)
                for pmf in probas
            ]
        )

    def accuracy_(self, probas: Tensor, batching: bool = True) -> Tensor:
        choices = self.break_ties(probas)
        argmax = np.where(choices == 0, 1, 0)
        acc = argmax.mean() if batching else argmax.tolist()
        return acc

    def choice_accuracy(self, similarities: float) -> float:
        probas = F.softmax(torch.stack(similarities, dim=-1), dim=1)
        choice_acc = self.accuracy_(probas)
        return choice_acc

    @staticmethod
    def unbind(embeddings: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        return torch.unbind(
            torch.reshape(embeddings, (-1, 3, embeddings.shape[-1])), dim=1
        )

    def regularization(self, alpha: float = 0.5) -> Tensor:
        """Apply combination of l2 and l1 regularization during training."""
        # NOTE: Frobenius norm is equivalent to torch.linalg.vector_norm(self.transform, ord=2, dim=(0, 1)))
        l2_reg = alpha * torch.linalg.norm(self.transform, ord="fro")
        l1_reg = (1 - alpha) * torch.linalg.vector_norm(
            self.transform, ord=1, dim=(0, 1)
        )
        complexity_loss = self.lmbda * (l2_reg + l1_reg)
        return complexity_loss

    def training_step(self, one_hots: Tensor, batch_idx: int):
        # training_step defines the train loop. It is independent of forward
        embedding = self.features @ self.transform
        embeddings = one_hots @ embedding
        anchor, positive, negative = self.unbind(embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        c_entropy = self.cross_entropy_loss(similarities)
        # apply l1 and l2 regularization during training to prevent overfitting to train objects
        complexity_loss = self.regularization()
        loss = c_entropy + complexity_loss
        acc = self.choice_accuracy(similarities)
        self.log("train_loss", c_entropy, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, one_hots: Tensor, batch_idx: int):
        embedding = self.features @ self.transform
        embeddings = one_hots @ embedding
        anchor, positive, negative = self.unbind(embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        val_loss = self.cross_entropy_loss(similarities)
        val_acc = self.choice_accuracy(similarities)
        self.log("val_loss", val_loss)
        self.log("val_acc", val_acc)
        return val_loss, val_acc

    def test_step(self, one_hots: Tensor, batch_idx: int):
        embedding = self.features @ self.transform
        embeddings = one_hots @ embedding
        anchor, positive, negative = self.unbind(embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        test_loss = self.cross_entropy_loss(similarities)
        test_acc = self.choice_accuracy(similarities)
        self.log("test_loss", test_loss)
        self.log("test_acc", test_acc)
        return test_loss, test_acc

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        if self.optim.lower() == "adam":
            optimizer = getattr(torch.optim, self.optim.capitalize())
        elif self.optim.lower() == "sgd":
            optimizer = getattr(torch.optim, self.optim.upper())
        else:
            raise ValueError(
                "\nUse Adam or SGD for learning a transformation of a network's feature space.\n"
            )
        optimizer = optimizer(self.parameters(), lr=self.lr)
        return optimizer
