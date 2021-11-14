MLFLOW_EXPERIMENT_NAME=cryptobot_model

.PHONY: test
test:
	poetry run python -m unittest

.PHONY: train
train:
	poetry run mlflow run . --experiment-name=$(MLFLOW_EXPERIMENT_NAME) --no-conda

.PHONY: ui
ui:
	poetry run mlflow ui
clearn_data: data_clearning/src/data_cleaning.py
	poetry run python data_clearning/src/data_cleaning.py \
		--n_jobs=5