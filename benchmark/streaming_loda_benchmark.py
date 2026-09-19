import sys
sys.path.append("../")

import argparse
import joblib
import logging
import os

from common import data_preprocess
from common.dataloader import load_dataset
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
from common.utils import seed_everything, load_config, set_logger, print_to_json
from networks.streaming_loda.models import StreamingLODA


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    logging.info("Saved StreamingLODA model to %s", file_path)


seed_everything()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./benchmark_config/", help="The config directory.")
    parser.add_argument("--expid", type=str, default="streaming_loda_SMD")
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

    # Preprocessing: fit scaler trên train và transform train/test giống các benchmark hiện tại.
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])
    pp.save(params["model_root"])

    evaluator = Evaluator(**params["eval"])

    for entity in params["entities"]:
        logging.info("Fitting dataset: %s", entity)

        train = data_dict[entity]["train"]
        test = data_dict[entity]["test"]
        test_label = data_dict[entity]["test_label"]

        model = StreamingLODA(
            n_features=train.shape[1],
            n_histograms=params.get("n_histograms", 100),
            histograms_per_batch=params.get("histograms_per_batch", 1),
            memory=params.get("memory", 5),
            score_decay=params.get("score_decay", 1.0),
            padding=params.get("padding", 0.1),
            epsilon=params.get("epsilon", 1.0e-10),
            standardize_window=params.get("standardize_window", True),
            random_state=params.get("seed", None),
        )

        tt = TimeTracker()

        # Streaming training: batch đầu dùng để khởi tạo model; các batch sau score trước rồi update.
        tt.train_start()
        train_anomaly_score = model.fit_stream(train, batch_size=params["batch_size"])
        tt.train_end()

        # Lưu state chỉ sau train, trước khi test update model, tránh lưu checkpoint đã nhìn thấy test data.
        entity_model_root = os.path.join(params["model_root"], entity)
        model_path = os.path.join(entity_model_root, "streaming_loda_model.joblib")
        save_model(model, model_path)

        # Prequential test: score batch trước, sau đó mới partial_fit batch đó nếu update_on_test=True.
        tt.test_start()
        anomaly_score = model.score_stream(test, batch_size=params["batch_size"], update=params.get("update_on_test", True))
        tt.test_end()

        store_entity(
            params,
            entity,
            train_anomaly_score,
            anomaly_score,
            test_label,
            time_tracker=tt.get_data(),
        )

        logging.info("StreamingLODA active histograms=%d/%d", model.n_active_histograms, model.n_histograms)

    evaluator.eval_exp(
        exp_folder=params["model_root"],
        entities=params["entities"],
        merge_folder=params["benchmark_dir"],
        extra_params=params,
        eval_single=True,
    )