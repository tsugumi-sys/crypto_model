import sys
import pandas as pd
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import logging
import argparse
from tqdm import tqdm
import os

sys.path.append(".")
from common.constants import DATAFOLDER, COINNAMES, COLUMNS
from common.custom_logger import CustomLogger
from common.progress_bar import custom_progressbar

logging.basicConfig(level=logging.INFO)
logger = CustomLogger("Data_Cleaning")


def data_clearning(coin_name: str):
    data_folder = os.path.join(DATAFOLDER.raw_data_root_path, coin_name)
    save_folder = os.path.join(DATAFOLDER.cleaned_data_root_path, coin_name)
    os.makedirs(save_folder, exist_ok=True)

    csv_files = os.listdir(data_folder)
    csv_files.sort()

    columns = COLUMNS.default_columns
    df = pd.DataFrame(columns=columns)
    df = df.set_index("OpenTime")

    for csv_file in csv_files:
        csv_file_path = os.path.join(data_folder, csv_file)
        csv_df = pd.read_csv(csv_file_path, names=columns, index_col="OpenTime")
        df = df.append(csv_df)

    df = sort_index_fillna(df, step=900)
    df.to_parquet(os.path.join(save_folder, f"{coin_name}.parquet.gzip"), engine="pyarrow", compression="gzip")


def sort_index_fillna(df: pd.DataFrame, step: int = 900):
    df = df.sort_index()
    df.index = df.index // 1000  # Convert milisecond to second.
    df = df.reindex(range(df.index[0], df.index[-1] + step, step))
    df = df.fillna(method="pad")

    print(f"The number of NaN is {df.isna().sum().sum()}. ({100*(df.isna().sum().sum()/len(df))}%)")
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
    coin_names = COINNAMES.coin_names()

    logger.info("Start clearning data ...")
    with custom_progressbar(tqdm(desc="Data Cleaning", total=len(coin_names))):
        Parallel(n_jobs=n_jobs)(
            delayed(data_clearning)(
                coin_name=coin_name,
            )
            for coin_name in coin_names
        )

    logger.info(f"Data has saved in {DATAFOLDER.cleaned_data_root_path}")


if __name__ == "__main__":
    main()
