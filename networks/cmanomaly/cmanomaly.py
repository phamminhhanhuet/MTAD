import logging
import os
import time

import numpy as np
import torch
from torch import nn

from common.utils import set_device


class CMAnomaly(nn.Module):
    """CMAnomaly model adapted from the original ISSRE'23 replication code."""

    def __init__(
        self,
        in_channels,
        window_size,
        dropout=0.0,
        prediction_length=1,
        prediction_dims=None,
        inter="FM",
        gamma=1.0,
        batch_size=1024,
        nb_epoch=100,
        lr=0.01,
        device="cpu",
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.window_size = int(window_size)
        self.prediction_length = int(prediction_length)
        self.prediction_dims = prediction_dims if prediction_dims else list(range(self.in_channels))
        self.inter = inter
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.nb_epoch = int(nb_epoch)
        self.lr = float(lr)
        self.device = set_device(device)

        final_output_dim = self.prediction_length * len(self.prediction_dims)

        if self.inter in ["TIME", "MEAN"]:
            clf_input_dim = self.in_channels
        elif self.inter == "DIM":
            clf_input_dim = self.window_size - 1
        elif self.inter == "CONCAT":
            clf_input_dim = self.in_channels * (self.window_size - 1)
        elif self.inter in ["FM_com", "FM"]:
            clf_input_dim = self.window_size - 1 + self.in_channels
        else:
            clf_input_dim = self.window_size - 1 + self.in_channels

        self.res_w = nn.Linear(
            self.in_channels * (self.window_size - 1),
            self.window_size - 1 + self.in_channels,
        )

        self.linear = nn.Sequential(
            nn.Linear(clf_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, final_output_dim),
        )

        self.dropout = nn.Dropout(dropout)
        self.loss_fn = nn.MSELoss(reduction="none")
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.001)

        self.best_metric = float("inf")
        self.worse_count = 0
        self.to(self.device)

    def FM_interaction(self, x):
        """Factorization-machine bi-interaction used by the original model."""
        sum_of_square = torch.sum(x, dim=1) ** 2
        square_of_sum = torch.sum(x ** 2, dim=1)
        return (sum_of_square - square_of_sum) * 0.5

    def forward(self, batch_window):
        self.batch_size_current = batch_window.size(0)

        x = batch_window[:, 0:-self.prediction_length, :]
        y = batch_window[:, -self.prediction_length:, self.prediction_dims]

        if self.inter == "FM_com":
            time_inter = self.FM_interaction(x)
            dim_inter = self.FM_interaction(x.transpose(2, 1))
            raw = self.res_w(x.reshape(self.batch_size_current, -1))
            inter = self.gamma * torch.cat([time_inter, dim_inter], dim=-1)
            outputs = raw + inter
        elif self.inter == "FM":
            time_inter = self.FM_interaction(x)
            dim_inter = self.FM_interaction(x.transpose(2, 1))
            outputs = torch.cat([time_inter, dim_inter], dim=-1)
        elif self.inter == "MEAN":
            outputs = x.mean(dim=1)
        elif self.inter == "TIME":
            outputs = self.FM_interaction(x)
        elif self.inter == "DIM":
            outputs = self.FM_interaction(x.transpose(2, 1))
        elif self.inter == "CONCAT":
            outputs = x.reshape(self.batch_size_current, -1)
        else:
            raise ValueError(f"Unsupported interaction mode: {self.inter}")

        outputs = self.dropout(outputs)
        recst = self.linear(outputs).view(
            self.batch_size_current,
            self.prediction_length,
            len(self.prediction_dims),
        )

        loss = self.loss_fn(recst, y)

        return {
            "loss": loss.mean(),
            "recst": recst,
            "repr": outputs,
            "score": loss,
            "y": y,
        }

    def fit(self, train_loader, checkpoint_path, patience=3):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        num_batches = len(train_loader)
        if num_batches == 0:
            raise ValueError("train_loader is empty")

        logging.info("Start CMAnomaly training for %d batches.", num_batches)
        train_start = time.time()

        for epoch in range(self.nb_epoch):
            self.train()
            running_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self.device).float()
                return_dict = self(batch)

                self.optimizer.zero_grad()
                loss = return_dict["loss"]
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / num_batches
            logging.info("Epoch: %d, loss: %.5f", epoch + 1, avg_loss)

            if avg_loss < self.best_metric:
                self.best_metric = avg_loss
                self.worse_count = 0
                self.save_checkpoint(checkpoint_path, epoch=epoch + 1, best_metric=avg_loss)
            else:
                self.worse_count += 1

            if self.worse_count >= patience:
                logging.info("Early stop at epoch %d.", epoch + 1)
                break

        logging.info("-- CMAnomaly training done in %ds.", int(time.time() - train_start))
        return self

    def predict_prob(self, data_loader):
        self.eval()
        score_list = []

        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device).float()
                return_dict = self(batch)

                # Keep the original CMAnomaly scoring rule:
                # sigmoid(element-wise MSE), then average over predicted dimensions.
                score = return_dict["score"].sigmoid().mean(dim=-1)
                score_list.append(score)

        scores = torch.cat(score_list, dim=0).cpu().numpy()

        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        return scores

    def save_checkpoint(self, file_path, epoch=None, best_metric=None):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_metric": self.best_metric if best_metric is None else best_metric,
        }

        torch.save(checkpoint, file_path)
        logging.info("Saved CMAnomaly checkpoint to %s", file_path)

    def load_checkpoint(self, file_path, load_optimizer=False):
        checkpoint = torch.load(file_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.load_state_dict(checkpoint["model_state_dict"])

            if load_optimizer and "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if checkpoint.get("best_metric") is not None:
                self.best_metric = checkpoint["best_metric"]
        else:
            # Compatibility with the original repository, which saved raw state_dict.
            self.load_state_dict(checkpoint)

        self.to(self.device)
        self.eval()
        logging.info("Loaded CMAnomaly checkpoint from %s", file_path)
        return checkpoint
