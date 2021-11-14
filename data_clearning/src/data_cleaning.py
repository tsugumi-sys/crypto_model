import sys
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import logging
import argparse
from tqdm import tqdm
import os

sys.path.append(".")
from common.constants import DATAFOLDER, ASSET_INFO
from common.custom_logger import CustomLogger
from common.progress_bar import custom_progressbar

logging.basicConfig(level=logging.INFO)
logger = CustomLogger("Data_Cleaning")


def data_clearning(data, asset_name, asset_id):
    coin_df = data[data["Asset_ID"] == asset_id].set_index("timestamp")
    coin_df = fillna(coin_df)
    asset_name = asset_name.replace(" ", "")
    coin_df.to_parquet(
        DATAFOLDER.cleaned_data_root_path + f"{asset_name}.parquet.gzip",
        engine="pyarrow",
        compression="gzip",
    )


def fillna(df: pd.DataFrame):
    df = df.sort_index()
    df = df.reindex(range(df.index[0], df.index[-1] + 60, 60), method="pad")
    return df


def main():
    parser = argparse.ArgumentParser(description="Data Clearning")
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=3,
        help="The number of cpu cores used for multiprocessing.",
    )
    args = parser.parse_args()
    n_jobs = args.n_jobs

    if cpu_count() < n_jobs:
        logger.warning(f"CPU core for multiproceeessing is set to {cpu_count()-1}.")
        n_jobs = cpu_count() - 1

    os.makedirs(DATAFOLDER.cleaned_data_root_path, exist_ok=True)

    logger.info("Loading raw data ...")
    data = pd.read_csv(DATAFOLDER.raw_data_root_path + "train.csv")
    asset_info = ASSET_INFO.asset_info

    logger.info("Start clearning data ...")
    with custom_progressbar(tqdm(desc="Data Cleaning", total=len(asset_info.keys()))):
        Parallel(n_jobs=n_jobs)(
            delayed(data_clearning)(
                data=data,
                asset_name=name,
                asset_id=asset_info[name]["id"],
            )
            for name in asset_info.keys()
        )

    logger.info(f"Data has saved in {DATAFOLDER.cleaned_data_root_path}")


if __name__ == "__main__":
    main()
