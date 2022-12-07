from typing import Any, List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from .triplet_loss import TripletLoss

FrozenDict = Any
Tensor = torch.Tensor


class Linear(pl.LightningModule):
    def __init__(
        self,
        features: Tensor,
        optim_cfg: FrozenDict,
    ):
        super().__init__()
        self.features = torch.nn.Parameter(
            torch.from_numpy(features),
            requires_grad=False,
        )
        self.feature_dim = self.features.shape[1]
        # initialize transformation matrix with \tau I (temperature-scaled identity matrix)
        self.transform = torch.nn.Parameter(
            data=torch.eye(self.feature_dim),  # * optim_cfg["temperature"],
            requires_grad=True,
        )
        self.optim = optim_cfg["optim"]
        self.lr = optim_cfg["lr"]
        self.lmbda = optim_cfg["lmbda"]

        self.loss_fun = TripletLoss(temperature=1.0)

    def forward(self, one_hots: Tensor) -> Tensor:
        embedding = self.features @ self.transform
        # normalize object embeddings to lie on the unit-sphere
        # embedding = F.normalize(embedding, dim=1)
        batch_embeddings = one_hots @ embedding
        return batch_embeddings

    @staticmethod
    def convert_predictions(sim_predictions: Tensor) -> Tensor:
        """Convert similarity predictions into odd-one-out predictions."""
        first_conversion = torch.where(
            sim_predictions != 1, sim_predictions - 2, sim_predictions
        )
        ooo_predictions = torch.where(first_conversion < 0, 2, first_conversion)
        return ooo_predictions

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

    @staticmethod
    def normalize(triplet: List[Tensor]) -> List[Tensor]:
        """Normalize object embeddings to have unit norm."""
        return list(map(lambda obj: F.normalize(obj, dim=1), triplet))

    def regularization(self, alpha: float = 1.0) -> Tensor:
        """Apply combination of l2 and l1 regularization during training."""
        # NOTE: Frobenius norm in PyTorch is equivalent to torch.linalg.vector_norm(self.transform, ord=2, dim=(0, 1)))
        l2_reg = alpha * torch.linalg.norm(self.transform, ord="fro")
        l1_reg = (1 - alpha) * torch.linalg.vector_norm(
            self.transform, ord=1, dim=(0, 1)
        )
        complexity_loss = self.lmbda * (l2_reg + l1_reg)
        return complexity_loss

    def training_step(self, one_hots: Tensor, batch_idx: int):
        batch_embeddings = self(one_hots)
        anchor, positive, negative = self.unbind(batch_embeddings)
        dots = self.compute_similarities(anchor, positive, negative)
        c_entropy = self.loss_fun(dots)
        # apply l1 and l2 regularization during training to prevent overfitting to train objects
        complexity_loss = self.regularization()
        loss = c_entropy + complexity_loss
        acc = self.choice_accuracy(dots)
        self.log("train_loss", c_entropy, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True)
        return loss

    def validation_step(self, one_hots: Tensor, batch_idx: int):
        loss, acc = self._shared_eval_step(one_hots, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics

    def test_step(self, one_hots: Tensor, batch_idx: int):
        loss, acc = self._shared_eval_step(one_hots, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics

    def _shared_eval_step(self, one_hots: Tensor, batch_idx: int):
        batch_embeddings = self(one_hots)
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        loss = self.loss_fun(similarities)
        acc = self.choice_accuracy(similarities)
        return loss, acc

    def predict_step(self, one_hots: Tensor, batch_idx: int):
        batch_embeddings = self(one_hots)
        anchor, positive, negative = self.unbind(batch_embeddings)
        similarities = self.compute_similarities(anchor, positive, negative)
        sim_predictions = torch.argmax(
            F.softmax(torch.stack(similarities, dim=1), dim=1), dim=1
        )
        ooo_predictions = self.convert_predictions(sim_predictions)
        return ooo_predictions

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()

    def configure_optimizers(self):
        if self.optim.lower() == "adam":
            optimizer = getattr(torch.optim, self.optim.capitalize())
            optimizer = optimizer(self.parameters(), lr=self.lr)
        elif self.optim.lower() == "sgd":
            optimizer = getattr(torch.optim, self.optim.upper())
            optimizer = optimizer(self.parameters(), lr=self.lr, momentum=0.9)
        else:
            raise ValueError(
                "\nUse Adam or SGD for learning a linear transformation of a network's feature space.\n"
            )
        return optimizer
