.PHONY: test
test:
	poetry run python -m unittest

.PHONY: clearn_data
clearn_data:
	poetry run python data_clearning/src/data_cleaning.py \
		--n_jobs=5