import pandas as pd
import os
from typing import Dict
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from src.features import sma, std
from src.label_features import position_label

sys.path.append("..")
from common.constants import DATAFOLDER, COLUMNS
from common.custom_logger import CustomLogger
from common.progress_bar import custom_progressbar

logger = CustomLogger("Preprocess_Logger")


def agg_cpcv_folds(n_jobs: int, downstream_directory: str, split_meta_info):
    if cpu_count() < n_jobs:
        logger.warning(f"CPU core for multiproceeessing is set to {cpu_count()-1}.")
        n_jobs = cpu_count() - 1

    with custom_progressbar(tqdm(desc="Aggregation preocess ...", total=len(list(split_meta_info.values())))):
        data_meta_info = Parallel(n_jobs=n_jobs)(
            delayed(agg_feature)(
                downstream_directory=downstream_directory,
                data_timestamps=data_timestamps,
            )
            for data_timestamps in split_meta_info.values()
        )

    return data_meta_info


def agg_feature(downstream_directory: str, data_timestamps: Dict):
    downstream_directory = os.path.abspath(downstream_directory)

    coin_name = data_timestamps["coin_name"]
    data_file_path = os.path.join(
        DATAFOLDER.cleaned_data_root_path,
        coin_name,
        f"{coin_name}.parquet.gzip",
    )
    df = pd.read_parquet(data_file_path, engine="pyarrow")

    target_columns = COLUMNS.target_columns
    feature_columns = []

    save_dir = os.path.join(downstream_directory, "cpcv_fold")
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, coin_name)
    os.makedirs(save_dir, exist_ok=True)

    for cpcv_fold_idx, cpcv_fold in enumerate(data_timestamps["cpcv_folds"]):
        for senario_idx in cpcv_fold.keys():
            for _type in ["train", "test", "valid"]:
                _save_dir = os.path.join(save_dir, f"fold_{cpcv_fold_idx}", f"senario_{senario_idx}", _type)
                os.makedirs(_save_dir, exist_ok=True)

                for _fold_idx in cpcv_fold[senario_idx][_type]:
                    start_idx, end_idx = cpcv_fold[senario_idx][_type][_fold_idx]["start"], cpcv_fold[senario_idx][_type][_fold_idx]["end"]
                    _df = df.loc[start_idx : end_idx + 1, target_columns]

                    # input features
                    _df["sma5"] = sma(_df, 5)
                    _df["sma25"] = sma(_df, 25)
                    _df["sma50"] = sma(_df, 50)
                    _df["sma100"] = sma(_df, 100)
                    feature_columns += [f"sma{i}" for i in [5, 25, 50, 100]]

                    _df["std10"] = std(_df, 10)
                    _df["std25"] = std(_df, 25)
                    _df["std50"] = std(_df, 50)
                    feature_columns += [f"std{i}" for i in [10, 25, 50]]

                    _df["sma5_20_deviation_rate"] = 100 * (1 - sma(_df, 20)) / sma(_df, 5)
                    _df["sma10_30_deviation_rate"] = 100 * (1 - sma(_df, 30)) / sma(_df, 10)
                    _df["sma25_50_deviation_rate"] = 100 * (1 - sma(_df, 50)) / sma(_df, 25)
                    feature_columns += [f"sma{i}_{j}_deviation_rate" for i, j in zip([5, 10, 25], [20, 30, 50])]

                    # labels
                    _df["y_buy"], _df["buy_cost"], _df["y_sell"], _df["sell_cost"] = position_label(
                        _df, pips=1.0, fee_percent=0.1, horizon_barrier=1, executionType="limit"
                    )
                    feature_columns += ["y_buy", "buy_cost", "y_sell", "sell_cost"]
                    _df = _df.dropna()
                    _df = _df[feature_columns]
                    feature_columns = []

                    save_path = os.path.join(_save_dir, f"{_fold_idx}.parquet.gzip")
                    _df.to_parquet(save_path, engine="pyarrow", compression="gzip")
                    data_timestamps["cpcv_folds"][cpcv_fold_idx][senario_idx][_type][_fold_idx] = save_path

    return data_timestamps
