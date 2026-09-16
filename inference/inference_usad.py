import sys
sys.path.append("../")

import argparse
import json
import logging
import pickle

import numpy as np
import torch

from common import data_preprocess
from common.utils import load_config, set_logger, print_to_json
from networks.usad import UsadModel


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


def build_model(params):
    w_size = params["window_size"] * params["dim"]
    z_size = params["window_size"] * params["hidden_size"]

    return UsadModel(w_size=w_size, z_size=z_size, device=params["device"])


def prepare_input(raw_input, pp, entity, window_size, dim):
    windows = np.asarray(raw_input, dtype=np.float32)
    windows = np.nan_to_num(windows, nan=0.0, posinf=0.0, neginf=0.0)

    single_input = False

    if windows.ndim == 2:
        if windows.shape != (window_size, dim):
            raise ValueError(f"Single input must have shape ({window_size}, {dim}), got {windows.shape}")

        windows = windows[None, ...]
        single_input = True

    elif windows.ndim == 3:
        if windows.shape[1:] != (window_size, dim):
            raise ValueError(f"Batch input must have shape (batch_size, {window_size}, {dim}), got {windows.shape}")

    else:
        raise ValueError(f"Input must be 2D or 3D, got shape={windows.shape}")

    windows = pp.transform(entity, windows)
    return windows, single_input


def predict(model, raw_input, pp, entity, window_size, dim, batch_size=256, alpha=0.5, beta=0.5):
    windows, single_input = prepare_input(
        raw_input=raw_input,
        pp=pp,
        entity=entity,
        window_size=window_size,
        dim=dim,
    )

    model.eval()
    model.to(model.device)

    anomaly_scores = []

    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            batch_windows = windows[start:start + batch_size]

            # USAD nhận mỗi window dưới dạng vector phẳng.
            batch = torch.from_numpy(batch_windows).float().reshape(-1, model.w_size).to(model.device)
            print("batch shape:", batch.shape)
            # Reconstruction từ Autoencoder 1.
            z = model.encoder(batch)
            w1 = model.decoder1(z)

            # Reconstruction adversarial qua Decoder 2.
            z1 = model.encoder(w1)
            w2 = model.decoder2(z1)

            reconstruction_error_1 = torch.mean((batch - w1) ** 2, dim=1)
            reconstruction_error_2 = torch.mean((batch - w2) ** 2, dim=1)

            score = alpha * reconstruction_error_1 + beta * reconstruction_error_2
            anomaly_scores.append(score.detach().cpu().numpy())

    anomaly_scores = np.concatenate(anomaly_scores, axis=0)
    print("anomaly_scores " , anomaly_scores)
    if single_input:
        return float(anomaly_scores[0])

    return anomaly_scores


def create_windows(data, window_size, stride=1):
    windows, _ = data_preprocess.get_windows(data, window_size=window_size, stride=stride)
    return windows


def main(args):
    params = load_config(args["config"], args["expid"])
    set_logger(params, args)
    logging.info(print_to_json(params))

    entity = args["entity"]

    print("Entity:", entity)
    print("Model path:", args["model_path"])

    # Build model và load checkpoint.
    model = build_model(params)
    model.load_checkpoint(args["model_path"])

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
        raw_windows = create_windows(raw_data, window_size=params["window_size"], stride=params["stride"])
        print("Generated windows:", raw_windows.shape)

        scores = predict(
            model=model,
            raw_input=raw_windows,
            pp=pp,
            entity=entity,
            window_size=params["window_size"],
            dim=params["dim"],
            batch_size=args["batch_size"],
            alpha=args["alpha"],
            beta=args["beta"],
        )

        labels = (scores >= threshold).astype(int)
        print("Anomarly index ", np.where(labels == 1)[0])

        print("Anomaly scores shape:", scores.shape)
        print("Score min:", scores.min())
        print("Score max:", scores.max())
        print("Score mean:", scores.mean())
        print("Detected abnormal windows:", labels.sum(), "/", len(labels))
        return

    # Infer một window cụ thể.
    index = args["index"]
    end_index = index + params["window_size"]

    if index < 0 or end_index > len(raw_data):
        raise ValueError(f"Invalid index={index}. Need {params['window_size']} points, but dataset length is {len(raw_data)}.")

    raw_window = raw_data[index:end_index]

    print("Window index:", index)
    print("Raw input shape:", raw_window.shape)

    score = predict(
        model=model,
        raw_input=raw_window,
        pp=pp,
        entity=entity,
        window_size=params["window_size"],
        dim=params["dim"],
        batch_size=args["batch_size"],
        alpha=args["alpha"],
        beta=args["beta"],
    )

    classification = "abnormal" if score >= threshold else "normal"

    print("Anomaly score:", score)
    print(f"Classification: {classification} (threshold={threshold})")


def parse_args():
    parser = argparse.ArgumentParser(description="USAD inference")

    parser.add_argument("--config", type=str, default="../benchmark/benchmark_config/", help="Config directory")
    parser.add_argument("--expid", type=str, default="usad_SMD", help="Experiment id")
    parser.add_argument("--entity", type=str, default="machine-1-1", help="Entity id")
    parser.add_argument("--model_path", type=str, required=True, help="Path to USAD checkpoint")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to raw input pkl")
    parser.add_argument("--threshold_file", type=str, required=True, help="Path to thresholds.json")
    parser.add_argument("--threshold_type", type=str, default="pot", choices=["pot", "best"], help="Threshold type")
    parser.add_argument("--index", type=int, default=0, help="Start index of a single inference window")
    parser.add_argument("--all", action="store_true", help="Run inference for all windows")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight of AE1 reconstruction error")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight of AE2 reconstruction error")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for CPU")

    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())

"""
python inference_usad.py \
    --expid usad_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/usad/usad_SMD/machine-1-1/usad_model.pt \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/usad/usad_SMD/machine-1-1/thresholds.json \
    --index 0 \
    --gpu 0

python inference_usad.py \
    --expid usad_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/usad/usad_SMD/machine-1-1/usad_model.pt \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/usad/usad_SMD/machine-1-1/thresholds.json \
    --index 20030 \
    --gpu 0

python inference_usad.py \
    --expid usad_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/usad/usad_SMD/machine-1-1/usad_model.pt \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/usad/usad_SMD/machine-1-1/thresholds.json \
    --all \
    --gpu 0
"""