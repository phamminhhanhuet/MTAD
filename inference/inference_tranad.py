import sys
sys.path.append("../")

import argparse
import logging
import numpy as np
import pickle

from common import data_preprocess
from common.utils import load_config, set_logger, print_to_json
from networks.tranad.models import TranAD



def load_pkl(filepath, dim=None):
    with open(filepath, "rb") as f:
        data = pickle.load(f)

    data = np.asarray(data)

    if dim is not None:
        data = data.reshape(-1, dim)

    return data

def load_json_threshold(filepath):
    import json

    with open(filepath, "r") as f:
        threshold_dict = json.load(f)

    return threshold_dict

def create_windows(data, window_size, stride=1):
    if data.ndim != 2:
        raise ValueError(
            f"Expected (T, dim), got {data.shape}"
        )

    windows = []

    for start in range(
        0,
        len(data) - window_size + 1,
        stride,
    ):
        windows.append(
            data[start:start + window_size]
        )

    return np.asarray(
        windows,
        dtype=np.float32,
    )



# raw_ts = np.random.rand(1000, 38)

# raw_windows = create_windows(
#     raw_ts,
#     window_size=10,
# )

# print(raw_windows.shape)

# results = predict(
#     model,
#     raw_windows,
#     pp,
#     entity,
#     window_size=10,
#     dim=38,
# )


def prepare_input(raw_input, pp, entity, window_size, dim):
    x = np.asarray(raw_input, dtype=np.float32)

    # giống preprocessing khi training
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.ndim == 2:
        if x.shape != (window_size, dim):
            raise ValueError(
                f"Single input must have shape "
                f"({window_size}, {dim}), got {x.shape}"
            )

    elif x.ndim == 3:
        if x.shape[1:] != (window_size, dim):
            raise ValueError(
                f"Batch input must have shape "
                f"(batch_size, {window_size}, {dim}), got {x.shape}"
            )

    else:
        raise ValueError(
            f"Input must be 2D or 3D, got shape={x.shape}"
        )

    return pp.transform(entity, x)


def predict(model, raw_input, pp, entity, window_size, dim):
    x = prepare_input(
        raw_input=raw_input,
        pp=pp,
        entity=entity,
        window_size=window_size,
        dim=dim,
    )

    # single window
    if x.ndim == 2:
        return model.predict_one(x)

    # batch windows
    results = []
    for sample in x:
        results.append(model.predict_one(sample))

    return np.asarray(results)


def main(args):
    params = load_config(args["config"], args["expid"])

    set_logger(params, args)
    logging.info(print_to_json(params))

    entity = args["entity"]

    model = TranAD(
        params["dim"],
        params["window_size"],
        lr=params["lr"],
        model_root=params["model_root"],
        device=params["device"],
    )

    model.load_checkpoint(args["model_path"])

    data_pkl = load_pkl(filepath=args["pkl_path"], dim=params["dim"])
    print("Loaded data from {} with shape {}".format(args["pkl_path"], data_pkl.shape))

    # load fitted scaler
    pp = data_preprocess.preprocessor(
        model_root=params["model_root"]
    )
    pp.load(params["model_root"])

    # load threshold
    threshold_dict = load_json_threshold(args["threshold_file"])

    #
    # Ví dụ 1: một window
    #
    # raw_input = np.random.rand(
    #     params["window_size"],
    #     params["dim"],
    # ).astype(np.float32)

    normal_index = 0
    print("Normal index ", normal_index)
    raw_normal_input = data_pkl[normal_index:normal_index + params["window_size"]]

    print("Raw normal input shape:", raw_normal_input.shape)
    print("Raw normal input:", raw_normal_input)

    normal_result = predict(
        model=model,
        raw_input=raw_normal_input,
        pp=pp,
        entity=entity,
        window_size=params["window_size"],
        dim=params["dim"],
    )

    print("Single input shape:", raw_normal_input.shape)
    print("Single prediction:", normal_result)

    cls = "abnormal" if normal_result >= float(threshold_dict["pot"]) else "normal"
    print(f"Classification: {cls} (threshold={threshold_dict['pot']})")

    #
    # Ví dụ 2: batch
    #
    # raw_batch = np.random.rand(
    #     32,
    #     params["window_size"],
    #     params["dim"],
    # ).astype(np.float32)

    abnormal_index = 15849
    print("Abnormal index ", abnormal_index)
    raw_abnormal_data = data_pkl[abnormal_index:abnormal_index + params["window_size"] ]

    abnormal_result = predict(
        model=model,
        raw_input=raw_abnormal_data,
        pp=pp,
        entity=entity,
        window_size=params["window_size"],
        dim=params["dim"],
    )

    print("Abnormal input shape:", raw_abnormal_data.shape)
    print("Abnormal predictions:", abnormal_result)

    cls = "abnormal" if abnormal_result >= float(threshold_dict["pot"]) else "normal"
    print(f"Classification for abnormal data: {cls} (threshold={threshold_dict['pot']})")

def parse_args():
    parser = argparse.ArgumentParser(description="TranAD inference")

    parser.add_argument("--config", type=str, default="configs/tranad_config.yaml", help="config file")
    parser.add_argument("--expid",type=str,default="tranad",    help="experiment id for the config",)
    parser.add_argument("--entity",type=str,default="machine-1-1", help="entity id")
    parser.add_argument("--model_path",type=str,default="checkpoints/tranad/machine-1-1/model.pt", help="path to the model checkpoint",)
    parser.add_argument("--pkl_path",type=str,default="machine-1-1.pkl", help="path to the pkl file",)
    parser.add_argument("--threshold_file",type=str,default="threshold.json", help="path to the threshold file",)
    parser.add_argument("--gpu",type=int,default=-1,help="GPU id to use, -1 for CPU",)

    return parser.parse_args()


if __name__ == "__main__":
    main(vars(parse_args()))