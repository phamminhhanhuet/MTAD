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
from networks.mtad_gat import MTAD_GAT


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
    return MTAD_GAT(
        n_features=params["dim"],
        window_size=params["window_size"],
        out_dim=params["dim"],
        kernel_size=params["kernel_size"],
        feat_gat_embed_dim=params["feat_gat_embed_dim"],
        time_gat_embed_dim=params["time_gat_embed_dim"],
        use_gatv2=params["use_gatv2"],
        gru_n_layers=params["gru_n_layers"],
        gru_hid_dim=params["gru_hid_dim"],
        forecast_n_layers=params["forecast_n_layers"],
        forecast_hid_dim=params["forecast_hid_dim"],
        recon_n_layers=params["recon_n_layers"],
        recon_hid_dim=params["recon_hid_dim"],
        dropout=params["dropout"],
        alpha=params["alpha"],
        device=params["device"],
    )


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


def predict(model, raw_input, pp, entity, window_size, dim, gamma, batch_size=256):
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
            batch = torch.from_numpy(windows[start:start + batch_size]).to(model.device)

            # Giống sliding_window_dataset(next_steps=1) khi training/evaluation.
            x = batch[:, :-1, :]
            y = batch[:, -1:, :]

            # Forecasting branch.
            y_hat, _ = model(x)
            if y_hat.ndim == 3:
                y_hat = y_hat.squeeze(1)

            # Reconstruction branch.
            recon_x = torch.cat((x[:, 1:, :], y), dim=1)
            _, window_recon = model(recon_x)
            recon_last = window_recon[:, -1, :]

            # Giữ đúng semantics của predict_prob() hiện tại.
            actual = x[:, -1, :]

            forecast_score = torch.sqrt((y_hat - actual) ** 2)
            recon_score = torch.sqrt((recon_last - actual) ** 2)
            score = (forecast_score + gamma * recon_score).mean(dim=1)

            anomaly_scores.append(score.detach().cpu().numpy())

    anomaly_scores = np.concatenate(anomaly_scores, axis=0)

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
    checkpoint = model.load_checkpoint(args["model_path"])
    print("Loaded checkpoint epoch:", checkpoint.get("epoch"))

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
            gamma=params["gamma"],
            batch_size=args["batch_size"],
        )

        labels = (scores >= threshold).astype(int)

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
        gamma=params["gamma"],
        batch_size=args["batch_size"],
    )

    classification = "abnormal" if score >= threshold else "normal"

    print("Anomaly score:", score)
    print(f"Classification: {classification} (threshold={threshold})")


def parse_args():
    parser = argparse.ArgumentParser(description="MTAD-GAT inference")

    parser.add_argument("--config", type=str, default="../benchmark/benchmark_config/", help="Config directory")
    parser.add_argument("--expid", type=str, default="mtad_gat_SMD", help="Experiment id")
    parser.add_argument("--entity", type=str, default="machine-1-1", help="Entity id")
    parser.add_argument("--model_path", type=str, required=True, help="Path to MTAD-GAT checkpoint")
    parser.add_argument("--pkl_path", type=str, required=True, help="Path to raw input pkl")
    parser.add_argument("--threshold_file", type=str, required=True, help="Path to thresholds.json")
    parser.add_argument("--threshold_type", type=str, default="pot", choices=["pot", "best"], help="Threshold type")
    parser.add_argument("--index", type=int, default=0, help="Start index of a single inference window")
    parser.add_argument("--all", action="store_true", help="Run inference for all windows")
    parser.add_argument("--batch_size", type=int, default=256, help="Inference batch size")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU id, -1 for CPU")

    return vars(parser.parse_args())


if __name__ == "__main__":
    main(parse_args())

"""
python inference_mtad_gat.py \
    --expid mtad_gat_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/mtad_gat/mtad_gat_SMD/machine-1-1/model.pt \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/mtad_gat/mtad_gat_SMD/machine-1-1/thresholds.json \
    --index 0 \
    --gpu 0

python inference_mtad_gat.py \
    --expid mtad_gat_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/mtad_gat/mtad_gat_SMD/machine-1-1/model.pt \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/mtad_gat/mtad_gat_SMD/machine-1-1/thresholds.json \
    --index 15849 \
    --gpu 0

python inference_mtad_gat.py \
    --expid mtad_gat_SMD \
    --entity machine-1-1 \
    --model_path ../benchmark/benchmark_exp_details/SMD/mtad_gat/mtad_gat_SMD/machine-1-1/model.pt \
    --pkl_path /mnt/c/Users/Hacha/Documents/VHTWork/Datasets/AnomalyDetection/processed_dataset/processed_SMD/machine-1-1_test.pkl \
    --threshold_file ../benchmark/benchmark_exp_details/SMD/mtad_gat/mtad_gat_SMD/machine-1-1/thresholds.json \
    --all \
    --gpu 0

"""