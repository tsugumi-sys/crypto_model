import argparse
import os
import sys

from src.model import cpcv_train, Model

sys.path.append("..")
from common.custom_logger import CustomLogger

logger = CustomLogger("Train_Logger")


def main():
    parser = argparse.ArgumentParser(
        description="Train Run",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="lgbc",
        help="lgbc, lgbr or rfc",
    )
    parser.add_argument(
        "--upstream",
        type=str,
        default="/data/preprocess",
        help="Train upstream",
    )
    parser.add_argument(
        "--downstream",
        type=str,
        default="/data/train",
        help="Train downstream",
    )
    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="/data/evaluate/",
        help="Evaluate downstream directory.",
    )

    args = parser.parse_args()

    model_type = args.model_type
    upstream_direcotry = args.upstream
    downstream_directory = args.downstream
    evaluate_downstream_directory = args.evaluate_downstream
    os.makedirs(downstream_directory, exist_ok=True)
    os.makedirs(evaluate_downstream_directory, exist_ok=True)

    model, params = Model(model_type=model_type)
    logger.info(params)
    params["n_jobs"] = 5
    cpcv_train(
        model_type=model_type,
        model=model,
        params=params,
        upstream_directory=upstream_direcotry,
        evaluate_downstream_directory=evaluate_downstream_directory,
    )

    logger.info("Training has Finished!")


if __name__ == "__main__":
    main()
