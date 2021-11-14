from typing import Dict, List
import pandas as pd
import os
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

sys.path.append("..")
from common.constants import DATAFOLDER, ASSET_INFO
from common.custom_logger import CustomLogger
from common.progress_bar import custom_progressbar

logger = CustomLogger(__name__)


# load dataset and split into 6 dataset.
# return splited index for each coins.

# Split train, test and valid dataset.
# 5 stolyes.
# train - test - valid
# 1: [2, 3, 4] - [5] - [0, 1]
def split_train_test_valid(data_timestamps: List[Dict]) -> Dict:



def split_data(n_splits: int, n_jobs: int) -> List[Dict]:
    if n_splits < 0:
        logger.error("n_splits should be more than 0")
        raise ValueError("Invalid n_splits input.")

    if cpu_count() < n_jobs:
        logger.warning(f"CPU core for multiproceeessing is set to {cpu_count()-1}.")
        n_jobs = cpu_count() - 1

    asset_info = ASSET_INFO.asset_info

    with custom_progressbar(tqdm(desc=f"Data splitting into {n_splits}", total=len(asset_info.keys()))):
        data_timestamps = Parallel(n_jobs=n_jobs)(
            delayed(splitted_data_timestamps)(
                n_splits=n_splits,
                asset_name=name,
                asset_id=asset_info[name]["id"],
            )
            for name in asset_info.keys()
        )

    return data_timestamps


def splitted_data_timestamps(n_splits: int, asset_name: str, asset_id: int) -> Dict:
    _asset_name = asset_name.replace(" ", "")
    data = pd.read_parquet(
        os.path.join(DATAFOLDER.cleaned_data_root_path, f"{_asset_name}.parquet.gzip"),
        engine="pyarrow",
    )
    _timestamps = data.index.values
    split_length = len(_timestamps) // n_splits

    splitted_timestamps = {}
    first_timestamps = 0
    for i in range(split_length, len(_timestamps), split_length):
        splitted_timestamps[i // split_length - 1] = _timestamps[first_timestamps:i]
        first_timestamps = i

    splitted_timestamps[len(_timestamps) // split_length - 1] = _timestamps[first_timestamps:]
    return {"asset_name": asset_name, "asset_id": asset_id, "splitted_timestamps": splitted_timestamps}
