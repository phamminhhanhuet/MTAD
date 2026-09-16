import sys
sys.path.append("../")

import argparse
import json
import logging
import pickle
import joblib

import numpy as np

from common import data_preprocess
from common.utils import load_config, set_logger, print_to_json


def load_pkl(filepath, dim=None):
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    data = np.asarray(data)
    if dim is not None:
        data = data.reshape(-1, dim)

    return data


def load_threshold(filepath, threshold_type="pot"):
    with open(filepath, "r") as f:
        threshold_dict = json.load(f)

    if threshold_type not in threshold_dict:
        raise KeyError(f"Threshold '{threshold_type}' not found in {filepath}. Available keys: {list(threshold_dict.keys())}")

    return float(threshold_dict[threshold_type])


def load_model(file_path):
    model = joblib.load(file_path)
    logging.info(f"Loaded IForest model from {file_path}")
    return model


def prepare_input(raw_input, pp, entity, dim):
    x = np.asarray(raw_input, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    single_input = False

    if x.ndim == 1:
        if x.shape[0] != dim:
            raise ValueError(f"Single input must have shape ({dim},), got {x.shape}")

        x = x.reshape(1, -1)
        single_input = True

    elif x.ndim == 2:
        if x.shape[1] != dim:
            raise ValueError(f"Batch input must have shape (batch_size, {dim}), got {x.shape}")

    else:
        raise ValueError(f"Input must be 1D or 2D, got shape={x.shape}")

    x = pp.transform(entity, x)
    return x, single_input


def predict(model, raw_input, pp, entity, dim):
    x, single_input = prepare_input(raw_input=raw_input, pp=pp, entity=entity, dim=dim)

    # PyOD IForest: score càng lớn thì mức độ bất thường càng cao.
    anomaly_scores = model.decision_function(x)
    anomaly_scores = np.asarray(anomaly_scores).reshape(-1)

    if single_input:
        return float(anomaly_scores[0])

    return anomaly_scores


def main(args):
    params = load_config(args["config"], args["expid"])
    set_logger(params, args)
    logging.info(print_to_json(params))

    entity = args["entity"]

    print("Entity:", entity)
    print("Model path:", args["model_path"])

    # Load fitted IForest.
    model = load_model(args["model_path"])

    # Load fitted scaler từ training.
    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    pp.load(params["model_root"])

    # Load threshold.
    threshold = load_threshold(args["threshold_file"], threshold_type=args["threshold_type"])
    print(f"Threshold ({args['threshold_type']}):", threshold)

    # Load raw time-series.
    raw_data = load_pkl(filepath=args["pkl_path"], dim=params["dim"])
    print("Loaded raw data:", args["pkl_path"], "shape:", raw_data.shape)

    # Infer toàn bộ time-series.
    if args["all"]:
        scores = predict(model=model, raw_input=raw_data, pp=pp, entity=entity, dim=params["dim"])
        labels = (scores >= threshold).astype(int)

        print("Anomaly scores shape:", scores.shape)
        print("Score min:", scores.min())
        print("Score max:", scores.max())
        print("Score mean:", scores.mean())
        print("Detected abnormal samples:", labels.sum(), "/", len(labels))
        return

    # Infer một timestamp cụ thể.
    index = args["index"]

    if index < 0 or index >= len(raw_data):
        raise ValueError(f"Invalid index={index}. Dataset length is {len(raw_data)}.")

    raw_input = raw_data[index]

    print("Sample index:", index)
    print("Raw input shape:", raw_input.shape)
    print("Raw input:", raw_input)

    score = predict(model=model, raw_input=raw_input, pp=pp, entity=entity, dim=params["dim"])
    classification = "abnormal" if score >= threshold else "normal"

    print("Anomaly score:", score)
    print(f"Classification: {classification} (threshold={threshold})")


def parse_args():
    parser = argparse.ArgumentParser(description="IForest inference")

    parser.add_argument("--config", type=str, default="../benchmark/benchmark_config/", help="Config directory")
    parser.add_argument("--expid", type=str, default="iforest_SMD", help="Experiment id")
    parser.add_argument("--entity", type=str, default="machine-1-1", help="Entity id")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fitted IForest model")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to raw input pkl")
    parser.add_argument("--threshold_file", type=str, required=True, help="Path to thresholds.json")
    parser.add_argument("--threshold_type", type=str, default="pot", choices=["pot", "best"], help="Threshold type")
    parser.add_argument("--index", type=int, default=0, help="Index of a single inference sample")
    parser.add_argument("--all", action="store_true", help="Run inference for all samples")
    parser.add_argument("--gpu", type=int, default=-1, help="Unused by IForest; kept for config compatibility")

    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())

"""
python inference_iforest.py \
    --expid iforest_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/iforest/iforest_SMD/machine-1-1/iforest_model.joblib \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/iforest/iforest_SMD/machine-1-1/thresholds.json \
    --index 0 \
    --gpu 0

python inference_iforest.py \
    --expid iforest_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/iforest/iforest_SMD/machine-1-1/iforest_model.joblib \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/iforest/iforest_SMD/machine-1-1/thresholds.json \
    --index 20030 \
    --gpu 0

python inference_iforest.py \
    --expid iforest_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/iforest/iforest_SMD/machine-1-1/iforest_model.joblib \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/iforest/iforest_SMD/machine-1-1/thresholds.json \
    --all \
    --gpu 0
"""

