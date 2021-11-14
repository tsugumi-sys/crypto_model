import os
import argparse

import mlflow


def main():
    parser = argparse.ArgumentParser(
        description="Mlflow Runner",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--preprocess_downstream",
        type=str,
        default="/data/preprocess",
        help="preprocess downstream directory",
    )
    parser.add_argument(
        "--preprocess_n_jobs",
        type=int,
        default=5,
        help="The number of CPUs to use for preprocessing.",
    )
    parser.add_argument(
        "--preprocess_n_splits",
        type=int,
        default=6,
        help="The number of partitions for CV.",
    )

    args = parser.parse_args()
    mlflow_experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", 0)

    with mlflow.start_run():
        preprocess_run = mlflow.run(
            uri="./preprocess",
            entry_point="preprocess",
            backend="local",
            parameters={
                "downstream": args.preprocess_downstream,
                "n_jobs": args.preprocess_n_jobs,
                "n_splits": args.preprocess_n_splits,
            },
            use_conda=False,
        )
        preprocess_run = mlflow.tracking.MlflowClient().get_run(preprocess_run.run_id)


if __name__ == "__main__":
    main()
