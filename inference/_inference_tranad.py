import sys

sys.path.append("../")

import logging
from networks.tranad import *
from common import data_preprocess
from common.dataloader import load_dataset, get_dataloaders
from common.utils import seed_everything, load_config, set_logger, print_to_json
from networks.tranad.models import TranAD
from common.evaluation import Evaluator, TimeTracker
from common.exp import store_entity
import argparse
import numpy as np


def main(args):
    config_dir = args["config"]
    experiment_id = args["expid"]

    params = load_config(config_dir, experiment_id)

    set_logger(params, args)
    logging.info(print_to_json(params))

    print("Param from config ", params)

    model = TranAD(
        params["dim"],
        params["window_size"],
        lr=params["lr"],
        model_root=params["model_root"],
        device=params["device"],
    )

    model.load_checkpoint(args["model_path"])

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

    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    data_dict = pp.normalize(data_dict, method=params["normalize"])

    # sliding windows
    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    entity = "machine-1-1"
    windows = window_dict[entity]
    train_windows = windows["train_windows"]
    test_windows = windows["test_windows"]
    test_labels = windows["test_label"]

    abnormaly_test_labels = np.where(test_labels == 1)

    print("number of train window ", len(train_windows))
    print("test_labels shape ", test_labels.shape)
    print("train windows shape ", train_windows.shape)
    print("abnormaly_test_labels ", abnormaly_test_labels)

    test_normal_input = train_windows[0]

    print("test normal data ", test_normal_input)
    print("test data shape ", test_normal_input.shape)

    result = model.predict_one(test_normal_input)

    print("result ", result)

    abnormal_idx = abnormaly_test_labels[0][0]
    print("abnormal idx ", abnormal_idx)

    test_abnormal_input = test_windows[abnormal_idx]

    print("test_abnormal_input ", test_abnormal_input)

    result = model.predict_one(test_abnormal_input)

    print("result ", result)


def parse_args():
    parser = argparse.ArgumentParser(description="TranAD inference")
    parser.add_argument(
        "--config", type=str, default="configs/tranad_config.yaml", help="config file"
    )
    parser.add_argument(
        "--expid", type=str, default="tranad", help="experiment id for the config"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/tranad/machine-1-1/model.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU id to use, -1 for CPU"
    )
    return parser.parse_args() 

if __name__ == "__main__":
    args = parse_args()
    args = vars(args)
    main(args)