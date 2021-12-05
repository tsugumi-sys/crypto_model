import os
import sys
import argparse
import logging
import json
import mlflow

from src.extract_data import split_data
from src.feature_aggregation import agg_cpcv_folds

sys.path.append("..")
from common.custom_logger import CustomLogger

logging.basicConfig(level=logging.INFO)
logger = CustomLogger("Preprocess_Logger")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/data/preprocess/",
        help="downstream directory.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=5,
        help="The number of CPU cores to use for multiprocessing.",
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=6,
        help="The number of data partions.",
    )

    args = parser.parse_args()

    downstream_directory = args.downstream
    os.makedirs(downstream_directory, exist_ok=True)

    n_jobs = args.n_jobs
    n_splits = args.n_splits

    split_meta_info = split_data(n_splits=n_splits, n_jobs=n_jobs)
    data_meta_info = agg_cpcv_folds(
        n_jobs=n_jobs,
        downstream_directory=downstream_directory,
        split_meta_info=split_meta_info,
    )

    meta_split_path = os.path.join(downstream_directory, "meta_split.json")
    meta_data_path = os.path.join(downstream_directory, "meta_data.json")

    with open(meta_split_path, "w") as f:
        json.dump(split_meta_info, f)

    with open(meta_data_path, "w") as f:
        json.dump(data_meta_info, f)

    mlflow.log_artifact(meta_split_path)
    mlflow.log_artifact(meta_data_path)
    logger.info(f"Meta info files have been saved in {downstream_directory}")


if __name__ == "__main__":
    main()
