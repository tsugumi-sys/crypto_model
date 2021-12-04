import pandas as pd
import os
from typing import Dict
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from src.features import sma
from src.label_features import sample_target

sys.path.append("..")
from common.constants import DATAFOLDER
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

    asset_name = data_timestamps["asset_name"].replace(" ", "")
    data_file_path = os.path.join(
        DATAFOLDER.cleaned_data_root_path,
        f"{asset_name}.parquet.gzip",
    )
    df = pd.read_parquet(data_file_path, engine="pyarrow")
    target_columns = ["Open", "High", "Low", "Close", "Volume"]
    save_dir = os.path.join(downstream_directory, "cpcv_fold")
    os.makedirs(save_dir, exist_ok=True)
    save_dir = os.path.join(save_dir, asset_name)
    os.makedirs(save_dir, exist_ok=True)

    for cpcv_fold_idx, cpcv_fold in enumerate(data_timestamps["cpcv_folds"]):
        for senario_idx in cpcv_fold.keys():
            for _type in ["train", "test", "valid"]:
                _save_dir = os.path.join(save_dir, f"fold_{cpcv_fold_idx}", f"senario_{senario_idx}", _type)
                os.makedirs(_save_dir, exist_ok=True)

                for _fold_idx in cpcv_fold[senario_idx][_type]:
                    start_idx, end_idx = cpcv_fold[senario_idx][_type][_fold_idx]["start"], cpcv_fold[senario_idx][_type][_fold_idx]["end"]
                    _df = df.loc[start_idx : end_idx + 1, target_columns]
                    _df["sma20"] = sma(_df, 10)
                    _df["target"] = sample_target(_df)
                    _df = _df.dropna()
                    save_path = os.path.join(_save_dir, f"{_fold_idx}.parquet.gzip")
                    _df.to_parquet(save_path, engine="pyarrow", compression="gzip")
                    data_timestamps["cpcv_folds"][cpcv_fold_idx][senario_idx][_type][_fold_idx] = save_path

    return data_timestamps
