from typing import Optional
import multiprocessing as mlp
import logging

logger = logging.getLogger("Train_Logger")


class LGBMParameters:
    def __init__(
        self,
        objective: str = "regression",
        boosting_type: str = "gbdt",
        learning_rate: float = 0.1,
        lambda_l1: float = 0.0,
        lambda_l2: float = 0.0,
        num_leaves: int = 32,
        feature_fraction: float = 1.0,
        bagging_fraction: float = 1.0,
        bagging_freq: int = 0,
        min_child_samples: int = 20,
        n_jobs: int = 1,
        seed: int = 1234,
    ) -> None:
        self.objective = objective
        self.boosting_type = boosting_type
        self.learning_rate = learning_rate
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        self.num_leaves = num_leaves
        self.feature_fraction = feature_fraction
        self.bagging_fraction = bagging_fraction
        self.bagging_freq = bagging_freq
        self.min_child_samples = min_child_samples

        if n_jobs > mlp.cpu_count():
            logger.warning(f"Too big n_jobs value. Set n_jobs to {mlp.cpu_count() - 1}. (Your input is {n_jobs})")
            n_jobs = mlp.cpu_count() - 1
        self.n_jobs = n_jobs
        self.seed = seed


class RFParameters:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        random_state: int = 1234,
        n_jobs: int = 1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        if n_jobs > mlp.cpu_count():
            logger.warning(f"Too big n_jobs value. Set n_jobs to {mlp.cpu_count() - 1}. (Your input is {n_jobs})")
            n_jobs = mlp.cpu_count() - 1
        self.n_jobs = n_jobs
