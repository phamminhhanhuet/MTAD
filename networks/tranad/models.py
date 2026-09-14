import logging
import math
import os
import torch
import torch.nn as nn
import numpy as np
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from .dlutils import (
    PositionalEncoding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
)
from common.utils import set_device


class TranAD(nn.Module):
    def __init__(self, feats, window_size, lr, model_root, device):
        super(TranAD, self).__init__()
        self.name = "TranAD"
        self.n_feats = feats
        self.n_window = window_size
        self.device = set_device(device)
        self.n = self.n_feats * self.n_window
        self.pos_encoder = PositionalEncoding(2 * feats, 0.1, self.n_window)
        encoder_layers = TransformerEncoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
        decoder_layers1 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
        decoder_layers2 = TransformerDecoderLayer(
            d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1
        )
        self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
        self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())

        self.init_model(lr, model_root)

    def encode(self, src, c, tgt):
        src = torch.cat((src, c), dim=2)
        src = src * math.sqrt(self.n_feats)
        src = self.pos_encoder(src)
        memory = self.transformer_encoder(src)
        tgt = tgt.repeat(1, 1, 2)
        return tgt, memory

    def forward(self, src, tgt):
        # Phase 1 - Without anomaly scores
        c = torch.zeros_like(src)
        x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
        # Phase 2 - With anomaly scores
        c = (x1 - src) ** 2
        x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
        return x1, x2

    def init_model(self, lr, model_root, retrain=True, test=False):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

        if os.path.exists(model_root) and (not retrain or test):
            logging.info("Loading pre-trained model")
            checkpoint = torch.load(os.path.join(model_root, "model.pt"))
            self.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        else:
            logging.info("Creating new model: TranAD")

        self.optimizer = optimizer
        self.scheduler = scheduler
        logging.info("Finish model initialization.")

    def fit(self, nb_epoch, dataloader, training=True):
        self.to(self.device)
        for epoch in range(1, nb_epoch + 1):
            mse_func = nn.MSELoss(reduction="none")
            n = epoch + 1
            l1s = []
            if training:
                logging.info("Training epoch: {}".format(epoch))
                for d in dataloader:
                    d = d.to(self.device)
                    local_bs = d.shape[0]
                    window = d.permute(1, 0, 2)
                    elem = window[-1, :, :].view(1, local_bs, self.n_feats)
                    z = self(window, elem)
                    l1 = (
                        mse_func(z, elem)
                        if not isinstance(z, tuple)
                        else (1 / n) * mse_func(z[0], elem)
                        + (1 - 1 / n) * mse_func(z[1], elem)
                    )
                    if isinstance(z, tuple):
                        z = z[1]
                    l1s.append(torch.mean(l1).item())
                    loss = torch.mean(l1)
                    self.optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                self.scheduler.step()
                logging.info("Epoch: {} finished.".format(epoch))

    def predict_prob(self, test_iterator, label_windows=None):
        mse_func = nn.MSELoss(reduction="none")
        loss_steps = []
        for d in test_iterator:
            d = d.to(self.device)
            bs = d.shape[0]
            window = d.permute(1, 0, 2)
            elem = window[-1, :, :].view(1, bs, self.n_feats)
            z = self(window, elem)
            if isinstance(z, tuple):
                z = z[1]
            loss = mse_func(z, elem)[0]
            loss_steps.append(loss.detach().cpu().numpy())
        anomaly_score = np.concatenate(loss_steps).mean(axis=1)
        if label_windows is None:
            return anomaly_score
        else:
            anomaly_label = (np.sum(label_windows, axis=1) >= 1) + 0
            return anomaly_score, anomaly_label


    def _prepare_batch(self, batch):
        batch = batch.to(self.device, non_blocking=True)
        batch_size = batch.shape[0]
        print("Prepare batch - batch shape ", batch.shape)
        window = batch.permute(1, 0, 2)
        print("Prepare batch - window shape ", window.shape)
        target = window[-1].view(1, batch_size, self.n_feats)
        return window, target


    def predict_scores(self, dataloader):
        """Sinh một anomaly score cho đúng timestep cuối của mỗi window."""
        self.eval()
        scores = []

        with torch.inference_mode():
            for batch in dataloader:
                window, target = self._prepare_batch(batch)
                _, x2 = self(window, target)
                # Mean reconstruction MSE trên toàn bộ feature của entity.
                batch_scores = ((x2 - target) ** 2)[0].mean(dim=1)
                scores.append(batch_scores.cpu().numpy())

        if not scores:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(scores).astype(np.float32, copy=False)


    def predict_one(self, window):
        """Inference batch=1 cho một window đã được normalize, dùng để benchmark production."""
        self.eval()
        if not torch.is_tensor(window):
            window = torch.as_tensor(window, dtype=torch.float32)

        if tuple(window.shape) != (self.n_window, self.n_feats):
            raise ValueError(
                f"window phải có shape {(self.n_window, self.n_feats)}, nhận {tuple(window.shape)}"
            )

        with torch.inference_mode():
            print("Step 0 - window shape ", window.shape)
            batch = window.unsqueeze(0)
            print("Step 1 - batch shape ", batch.shape)
            src, target = self._prepare_batch(batch)
            print("Step 2 - src shape ", src.shape)
            print("Step 3 - target shape ", target.shape)
            _, x2 = self(src, target)
            return float((((x2 - target) ** 2).mean()).item())


    def save_checkpoint(self, checkpoint_path, epoch=None, entity=None):
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "epoch": epoch,
            "entity": entity,
            "n_feats": self.n_feats,
            "window_size": self.n_window,
        }

        torch.save(checkpoint, checkpoint_path)
        logging.info("Saved TranAD checkpoint to %s", checkpoint_path)


    def load_checkpoint(self, path, load_optimizer=True):
        # `weights_only` chỉ có ở PyTorch mới; fallback để chạy được cả version cũ.
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            checkpoint = torch.load(path, map_location=self.device)

        print("Load PARAM from checkpoint ")
        print("n_window ", checkpoint.get("window_size"), "self.n_window ", self.n_window)
        print("n_feats ", checkpoint.get("n_feats"), "self.n_feats ", self.n_feats)

        if checkpoint.get("n_feats") != self.n_feats:
            raise ValueError("Checkpoint có số feature khác model hiện tại")
        if checkpoint.get("window_size") != self.n_window:
            raise ValueError("Checkpoint có window_size khác model hiện tại")

        print("Load checkpoint for entity ", checkpoint["entity"])

        self.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.completed_epochs = int(checkpoint.get("completed_epochs", 0))
        self.to(self.device)
        self.eval()