
import sys


sys.path.append("../")

import argparse
import joblib
import json
import logging
import os
import pickle

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
    logging.info("Loaded StreamingLODA model from %s", file_path)
    return model


def save_model(model, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    logging.info("Saved updated StreamingLODA model to %s", file_path)


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
            raise ValueError(f"Input must have shape (N, {dim}), got {x.shape}")
    else:
        raise ValueError(f"Input must be 1D or 2D, got shape={x.shape}")

    x = pp.transform(entity, x)
    return x, single_input


def predict(model, raw_input, pp, entity, dim, batch_size=1000, update=True):
    x, single_input = prepare_input(raw_input=raw_input, pp=pp, entity=entity, dim=dim)

    if not model.is_fitted:
        raise RuntimeError("StreamingLODA has not been fitted")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    anomaly_scores = np.empty(x.shape[0], dtype=np.float64)

    # Prequential inference: mỗi batch được score bằng state hiện tại trước khi batch đó update model.
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        batch = x[start:end]
        anomaly_scores[start:end] = model.decision_function(batch)
        if update:
            model.partial_fit(batch)

    if single_input:
        return float(anomaly_scores[0])

    return anomaly_scores


def main(args):
    params = load_config(args["config"], args["expid"])
    set_logger(params, args)
    logging.info(print_to_json(params))

    entity = args["entity"]
    update_on_test = params.get("update_on_test", True) and not args["no_update"]

    print("Entity:", entity)
    print("Model path:", args["model_path"])
    print("Update after scoring:", update_on_test)

    model = load_model(args["model_path"])

    pp = data_preprocess.preprocessor(model_root=params["model_root"])
    pp.load(params["model_root"])

    threshold = load_threshold(args["threshold_file"], threshold_type=args["threshold_type"])
    print(f"Threshold ({args['threshold_type']}):", threshold)

    raw_data = load_pkl(filepath=args["pkl_path"], dim=params["dim"])
    print("Loaded raw data:", args["pkl_path"], "shape:", raw_data.shape)

    if args["all"]:
        scores = predict(
            model=model,
            raw_input=raw_data,
            pp=pp,
            entity=entity,
            dim=params["dim"],
            batch_size=params.get("batch_size", 1000),
            update=update_on_test,
        )

        labels = (scores >= threshold).astype(int)

        print("Anomaly scores shape:", scores.shape)
        print("Score min:", scores.min())
        print("Score max:", scores.max())
        print("Score mean:", scores.mean())
        print("Detected abnormal samples:", labels.sum(), "/", len(labels))

        if args["updated_model_path"] and update_on_test:
            save_model(model, args["updated_model_path"])
        return

    index = args["index"]
    if index < 0 or index >= len(raw_data):
        raise ValueError(f"Invalid index={index}. Dataset length is {len(raw_data)}.")

    # Với online/prequential mode, score tại index phụ thuộc các batch trước đó vì model được update liên tục.
    # Vì vậy replay stream từ đầu đến index để giữ đúng state tại thời điểm cần kiểm tra.
    if update_on_test:
        inference_data = raw_data[:index + 1]
        scores = predict(
            model=model,
            raw_input=inference_data,
            pp=pp,
            entity=entity,
            dim=params["dim"],
            batch_size=params.get("batch_size", 1000),
            update=True,
        )
        score = float(scores[-1])
    else:
        score = predict(
            model=model,
            raw_input=raw_data[index],
            pp=pp,
            entity=entity,
            dim=params["dim"],
            batch_size=params.get("batch_size", 1000),
            update=False,
        )

    classification = "abnormal" if score >= threshold else "normal"

    print("Sample index:", index)
    print("Anomaly score:", score)
    print(f"Classification: {classification} (threshold={threshold})")


def parse_args():
    parser = argparse.ArgumentParser(description="Streaming LODA inference")
    parser.add_argument("--config", type=str, default="../benchmark/benchmark_config/", help="Config directory")
    parser.add_argument("--expid", type=str, default="streaming_loda_SMD", help="Experiment id")
    parser.add_argument("--entity", type=str, default="machine-1-1", help="Entity id")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fitted StreamingLODA model")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to raw input pkl")
    parser.add_argument("--threshold_file", type=str, required=True, help="Path to thresholds.json")
    parser.add_argument("--threshold_type", type=str, default="pot", choices=["pot", "best"], help="Threshold type")
    parser.add_argument("--index", type=int, default=0, help="Index of a single inference sample")
    parser.add_argument("--all", action="store_true", help="Run inference for the complete stream")
    parser.add_argument("--no_update", action="store_true", help="Score without updating the StreamingLODA state")
    parser.add_argument("--updated_model_path", type=str, default=None, help="Optional path to save model state after --all inference")
    parser.add_argument("--gpu", type=int, default=-1, help="Unused by StreamingLODA; kept for config compatibility")
    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())


"""
python inference_streaming_loda.py \
    --config ../benchmark/benchmark_config/ \
    --expid streaming_loda_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/streaming_loda/streaming_loda_SMD/machine-1-1/streaming_loda_model.joblib \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/streaming_loda/streaming_loda_SMD/machine-1-1/thresholds.json \
    --all

python inference_streaming_loda.py \
    --config ../benchmark/benchmark_config/ \
    --expid streaming_loda_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/streaming_loda/streaming_loda_SMD/machine-1-1/streaming_loda_model.joblib \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/streaming_loda/streaming_loda_SMD/machine-1-1/thresholds.json \
    --index 15849

"""