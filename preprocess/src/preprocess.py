import os
import sys
import argparse
import logging

from src.extract_data import split_data

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

    splitted_data_info = split_data(n_splits=n_splits, n_jobs=n_jobs)
    print(splitted_data_info)

    logger.info(f"Meta info files have been saved in {downstream_directory}")


if __name__ == "__main__":
    main()
