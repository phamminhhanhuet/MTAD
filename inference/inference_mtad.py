import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

import sys
sys.path.append("../")

import logging
import argparse

from common import data_preprocess
from common.dataloader import load_dataset, get_dataloaders
from common.utils import seed_everything, load_config, set_logger, print_to_json
from networks.mtad_gat import MTAD_GAT


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


def main(args):
    params = load_config(args["config"], args["expid"])

    set_logger(params, args)
    logging.info(print_to_json(params))

    entity = args["entity"]

    model = build_model(params)
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

    window_dict = data_preprocess.generate_windows(
        data_dict,
        window_size=params["window_size"],
        stride=params["stride"],
    )

    windows = window_dict[entity]
    train_windows = windows["train_windows"]
    test_windows = windows["test_windows"]

    train_loader, _, test_loader = get_dataloaders(
        train_windows,
        test_windows,
        next_steps=1,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=params["num_workers"],
    )

    anomaly_score, anomaly_label = model.predict_prob(
        test_loader,
        gamma=params["gamma"],
        window_labels=windows["test_label"],
    )

    print("anomaly_score shape:", anomaly_score.shape)
    print("anomaly_label shape:", anomaly_label.shape)
    print("first anomaly scores:", anomaly_score[:20])


if __name__ == "__main__":
    seed_everything()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../benchmark/benchmark_config/")
    parser.add_argument("--expid", type=str, default="mtad_gat_SMD")
    parser.add_argument("--entity", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)

    args = vars(parser.parse_args())
    main(args)