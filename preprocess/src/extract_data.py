from typing import Dict, List
import pandas as pd
import numpy as np
import os
import sys
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

sys.path.append("..")
from common.constants import DATAFOLDER, COINNAMES
from common.custom_logger import CustomLogger
from common.progress_bar import custom_progressbar

logger = CustomLogger("Preprocess_Logger")


def split_data(n_splits: int, n_jobs: int) -> List[Dict]:
    if n_splits < 0:
        logger.error("n_splits should be more than 0")
        raise ValueError("Invalid n_splits input.")

    if cpu_count() < n_jobs:
        logger.warning(f"CPU core for multiproceeessing is set to {cpu_count()-1}.")
        n_jobs = cpu_count() - 1

    coin_names = COINNAMES.coin_names()

    with custom_progressbar(tqdm(desc=f"Data splitting into {n_splits}", total=len(coin_names))):
        all_data_timestamps = Parallel(n_jobs=n_jobs)(
            delayed(splitted_data_timestamps)(
                n_splits=n_splits,
                coin_name=coin_name,
            )
            for coin_name in coin_names
        )

    split_meta_info = {}
    for data_timestamps in all_data_timestamps:
        split_meta_info[data_timestamps["coin_name"]] = cpcv_split(n_test_folds=2, data_timestamps=data_timestamps)

    return split_meta_info


# Split train, test and valid dataset for CPCV.
def cpcv_split(n_test_folds: int, data_timestamps: List[Dict]) -> Dict:
    folds = list(data_timestamps["splitted_timestamps"].keys())
    n_folds = len(folds)

    if n_test_folds > n_folds:
        raise ValueError(f"n_test_folds: {n_test_folds} must be smaller than n_folds(n_splits): {n_folds}")

    selected_fold_bounds = list(itertools.combinations(folds, n_test_folds))
    num_senarios = int(len(selected_fold_bounds) / (n_folds / n_test_folds))

    cpcv_folds = []
    for i in range(num_senarios):
        _jump_folds = list(range(num_senarios - 1, 0, -1))[:i]

        while len(_jump_folds) < num_senarios - 1:
            _jump_folds.append(1)

        test_folds = []
        test_folds.append(i)
        for jump_idx, jump_val in enumerate(_jump_folds):
            test_folds.append(test_folds[jump_idx] + jump_val)

        if i == num_senarios - 1:
            # Last senario is mirror type of the first senario.
            # flip left-right side
            _senario = {}
            for key in list(cpcv_folds[0].keys())[::-1]:
                _senario[num_senarios - 1 - key] = {"train": {}, "valid": {}, "test": {}}
                for _type in ["train", "valid", "test"]:
                    for fold_idx in [n_folds - 1 - x for x in cpcv_folds[0][key][_type]]:
                        _senario[num_senarios - 1 - key][_type][fold_idx] = data_timestamps["splitted_timestamps"][fold_idx]
            cpcv_folds.append(_senario)
        else:
            _senario = {}
            _selected_valid_folds = []
            _selected_test_folds = []
            for sub_senario_idx, test_fold in enumerate(test_folds):
                _senario[sub_senario_idx] = {"train": {}, "valid": {}, "test": {}}

                # select valid set.
                for _valid_fold_idx in selected_fold_bounds[test_fold]:
                    if _valid_fold_idx not in _selected_valid_folds:
                        _senario[sub_senario_idx]["valid"][_valid_fold_idx] = data_timestamps["splitted_timestamps"][_valid_fold_idx]
                        _selected_valid_folds.append(_valid_fold_idx)

                # select test set.
                top_valid_fold = sorted(selected_fold_bounds[test_fold])[::-1][0]
                _test_fold_idx = top_valid_fold + 1 if top_valid_fold < num_senarios else 0
                while True:
                    if _test_fold_idx not in selected_fold_bounds[test_fold] and _test_fold_idx not in _selected_test_folds:
                        _senario[sub_senario_idx]["test"][_test_fold_idx] = data_timestamps["splitted_timestamps"][_test_fold_idx]
                        _selected_test_folds.append(_test_fold_idx)
                        break
                    _test_fold_idx += 1
                    if _test_fold_idx > num_senarios:
                        _test_fold_idx = 0

                # select train set.
                _selected_folds = [_test_fold_idx]
                _selected_folds += selected_fold_bounds[test_fold]
                for _train_fold_idx in [v for v in range(n_folds) if v not in _selected_folds]:
                    _senario[sub_senario_idx]["train"][_train_fold_idx] = data_timestamps["splitted_timestamps"][_train_fold_idx]
            cpcv_folds.append(_senario)

    return {"coin_name": data_timestamps["coin_name"], "cpcv_folds": cpcv_folds}


def splitted_data_timestamps(n_splits: int, coin_name: str) -> Dict:
    data = pd.read_parquet(
        os.path.join(DATAFOLDER.cleaned_data_root_path, coin_name, f"{coin_name}.parquet.gzip"),
        engine="pyarrow",
    )
    _timestamps = data.index.values.tolist()

    splitted_timestamps = {}
    for idx, timestamps in enumerate(np.array_split(np.array(_timestamps), n_splits)):
        splitted_timestamps[idx] = {
            "start": int(timestamps[0]),
            "end": int(timestamps[-1]),
        }
    return {"coin_name": coin_name, "splitted_timestamps": splitted_timestamps}
