import sys
sys.path.append("../")

import os
import logging
import argparse
import joblib

from common import data_preprocess
from common.dataloader import load_dataset
from common.utils import seed_everything, load_config, set_logger, print_to_json
from common.exp import store_entity
from common.evaluation import Evaluator, TimeTracker
from pyod.models.loda import LODA


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    logging.info(f"Saved LODA model to {file_path}")


def load_model(file_path):
    model = joblib.load(file_path)
    logging.info(f"Loaded LODA model from {file_path}")
    return model


seed_everything()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./benchmark_config/", help="The config directory.")
    parser.add_argument("--expid", type=str, default="loda_SMD")
    parser.add_argument("--gpu", type=int, default=-1)
    args = vars(parser.parse_args())

    config_dir = args["config"]
    experiment_id = args["expid"]

    params = load_config(config_dir, experiment_id)
    set_logger(params, args)
    logging.info(print_to_json(params))

    data_dict = load_dataset(
        data_root=params["data_root"],
        entities=params["entities"],
        dim=params["dim"],
        valid_ratio=params["valid_ratio"],
        test_label_postfix=params["test_label_postfix"],
        test_postfix=params["test_postfix"],
        train_postfix=params["train_postfix"],
        nrows=params["nrows"],
    )

    # Preprocessing
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # normalize() đã fit scaler cho từng entity, lưu lại để inference dùng đúng scaler training.
    pp.save(params["model_root"])

    evaluator = Evaluator(**params["eval"])

    for entity in params["entities"]:
        logging.info(f"Fitting dataset: {entity}")

        train = data_dict[entity]["train"]
        test = data_dict[entity]["test"]
        test_label = data_dict[entity]["test_label"]

        model = LODA(n_bins=params["n_bins"], n_random_cuts=params["n_random_cuts"])

        tt = TimeTracker()

        tt.train_start()
        model.fit(train)
        tt.train_end()

        entity_model_root = os.path.join(params["model_root"], entity)
        model_path = os.path.join(entity_model_root, "loda_model.joblib")
        save_model(model, model_path)

        train_anomaly_score = model.decision_function(train)

        tt.test_start()
        anomaly_score = model.decision_function(test)
        tt.test_end()

        anomaly_label = test_label

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