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

    parser.add_argument(
        "--train_model_type",
        type=str,
        default="lgbc",
        help="train model type.",
    )
    parser.add_argument(
        "--train_downstream",
        type=str,
        default="/data/train/",
        help="train downstream directory.",
    )

    parser.add_argument(
        "--evaluate_downstream",
        type=str,
        default="/data/evaluate/",
        help="Evaluate downstream direcotry.",
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

        # current_dir = os.getcwd()
        # train_upstream = os.path.join(
        #     current_dir,
        #     "mlruns/",
        #     str(mlflow_experiment_id),
        #     preprocess_run.info.run_id,
        #     "artifacts/",
        # )

        # train_upstream = "/Users/akiranoda/projects/crypto_model/mlruns/2/454431524cfa4334a86672a99b84b061/artifacts/"

        # train_run = mlflow.run(
        #     uri="./train",
        #     entry_point="train",
        #     backend="local",
        #     parameters={
        #         "model_type": args.train_model_type,
        #         "upstream": train_upstream,
        #         "downstream": args.train_downstream,
        #         "evaluate_downstream": args.evaluate_downstream,
        #     },
        #     use_conda=False,
        # )


if __name__ == "__main__":
    main()
