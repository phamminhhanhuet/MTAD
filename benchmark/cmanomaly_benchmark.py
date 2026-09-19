import sys
sys.path.append("../")

import os
import logging
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.exp import store_entity
from common.evaluation import Evaluator, TimeTracker
from networks.cmanomaly import CMAnomaly


seed_everything()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./benchmark_config/", help="The config directory.")
    parser.add_argument("--expid", type=str, default="cmanomaly_SMD")
    parser.add_argument("--gpu", type=int, default=-1)
    args = vars(parser.parse_args())

    params = load_config(args["config"], args["expid"])
    set_logger(params, args)
    logging.info(print_to_json(params))

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        valid_ratio=params["valid_ratio"],
        dim=params["dim"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        nrows=params["nrows"],
    )

    # Shared benchmark preprocessing. The fitted scaler is persisted for inference.
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # The original CMAnomaly preprocessing used MinMaxScaler(clip=True).
    # Preserve that behavior when the shared preprocessor exposes fitted scalers.
    if params["normalize"] == "minmax" and hasattr(pp, "scaler_dict"):
        for scaler in pp.scaler_dict.values():
            if hasattr(scaler, "clip"):
                scaler.clip = True

        # pp.normalize() has already transformed the arrays. The original CMAnomaly
        # uses MinMaxScaler(clip=True), so clip those transformed values directly.
        for entity in params["entities"]:
            data_dict[entity]["train"] = np.clip(data_dict[entity]["train"], 0.0, 1.0)
            data_dict[entity]["test"] = np.clip(data_dict[entity]["test"], 0.0, 1.0)

    os.makedirs(params["model_root"], exist_ok=True)
    pp.save(params["model_root"])

    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    evaluator = Evaluator(**params["eval"])

    for entity in params["entities"]:
        logging.info("Fitting dataset: %s", entity)

        windows = window_dict[entity]
        train_windows = windows["train_windows"]
        test_windows = windows["test_windows"]
        test_label_windows = windows["test_label"]

        train_loader = DataLoader(
            torch.from_numpy(train_windows).float(),
            batch_size=params["batch_size"],
            shuffle=True,
            num_workers=params.get("num_workers", 0),
        )

        train_score_loader = DataLoader(
            torch.from_numpy(train_windows).float(),
            batch_size=params.get("test_batch_size", 4096),
            shuffle=False,
            num_workers=params.get("num_workers", 0),
        )

        test_loader = DataLoader(
            torch.from_numpy(test_windows).float(),
            batch_size=params.get("test_batch_size", 4096),
            shuffle=False,
            num_workers=params.get("num_workers", 0),
        )

        model = CMAnomaly(
            in_channels=params["dim"],
            window_size=params["window_size"],
            dropout=params["dropout"],
            prediction_length=params["prediction_length"],
            prediction_dims=params.get("prediction_dims", []),
            inter=params["inter"],
            gamma=params["gamma"],
            batch_size=params["batch_size"],
            nb_epoch=params["nb_epoch"],
            lr=params["lr"],
            device=params["device"],
        )

        entity_model_root = os.path.join(params["model_root"], entity)
        checkpoint_path = os.path.join(entity_model_root, "cmanomaly_model.pt")

        tt = TimeTracker(nb_epoch=params["nb_epoch"])

        tt.train_start()
        model.fit(train_loader, checkpoint_path=checkpoint_path, patience=params["patience"])
        tt.train_end()

        # Match the original benchmark: evaluate with the best training-loss checkpoint.
        model.load_checkpoint(checkpoint_path)

        train_anomaly_score = model.predict_prob(train_score_loader)

        tt.test_start()
        anomaly_score = model.predict_prob(test_loader)
        tt.test_end()

        # Original CMAnomaly uses the label of the predicted endpoint, not "any anomaly in window".
        prediction_length = params["prediction_length"]
        if prediction_length == 1:
            anomaly_label = test_label_windows[:, -1]
        else:
            anomaly_label = (test_label_windows[:, -prediction_length:].sum(axis=1) > 0).astype(int)

        store_entity(
            params,
            entity,
            train_anomaly_score,
            anomaly_score,
            anomaly_label,
            time_tracker=tt.get_data(),
        )

    evaluator.eval_exp(
        exp_folder=params["model_root"],
        entities=params["entities"],
        merge_folder=params["benchmark_dir"],
        extra_params=params,
    )
