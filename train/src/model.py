from typing import Dict
import logging
import json
import os
import pandas as pd

import lightgbm as lgb
from lightgbm import early_stopping
from sklearn.ensemble import RandomForestClassifier
from src.parameters import LGBMParameters, RFParameters

logger = logging.getLogger("Train_Logger")


def Model(model_type: str):
    if model_type not in ["lgbc", "lgbr", "rfg"]:
        logger.error(f"{model_type} not in [lgbc, lgbr, rfg]")
        raise ValueError("Invalid model_type")

    if model_type in ["lgbc", "lgbr"]:
        objective = "regression" if model_type == "lgbr" else "binary"
        params = LGBMParameters(objective=objective)
        return LGBM(), params.__dict__

    elif model_type == "rfc":
        params = RFParameters()
        return RFClassifier(params.__dict__), params.__dict__


def LGBM():
    return lgb


def RFClassifier(params: Dict):
    return RandomForestClassifier(params)


def train_and_evaluate(
    model_type: str,
    model,
    params: Dict,
    train_dataset,
    test_dataset,
    evaluate_info,
):
    trained_model = None
    if model_type == "rfc":
        X_train, y_train = train_dataset.drop("target", axis=1), train_dataset["target"]
        trained_model = model.fit(X_train, y_train)

    elif model_type in ["lgbc", "lgbr"]:
        X_train, y_train = train_dataset.drop("target", axis=1), train_dataset["target"]
        X_test, y_test = test_dataset.drop("target", axis=1), test_dataset["target"]

        train_data = lgb.Dataset(X_train, y_train)
        test_data = lgb.Dataset(X_test, y_test, reference=train_data)

        trained_model = model.train(params, train_data, num_boost_round=200, valid_sets=[test_data], callbacks=[early_stopping(stopping_rounds=20)])

    for e_key in evaluate_info.keys():
        evaluate_dataset = evaluate_info[e_key]["data"]
        preds = trained_model.predict(evaluate_dataset.drop("target", axis=1).values)
        evaluate_dataset["predict"] = preds
        evaluate_dataset.to_parquet(evaluate_info[e_key]["filename"], engine="pyarrow", compression="gzip")


def cpcv_train(
    model_type: str,
    model,
    params: Dict,
    upstream_directory: str,
    evaluate_downstream_directory: str,
):
    meta_cpcv_info_path = os.path.join(upstream_directory, "meta_data.json")
    meta_cpcv_info = open(meta_cpcv_info_path)
    meta_cpcv_info = json.load(meta_cpcv_info)

    for each_asset in meta_cpcv_info:
        asset_name = each_asset["asset_name"]
        cpcv_folds = each_asset["cpcv_folds"]
        if asset_name == "Bitcoin":
            for fold_idx, fold in enumerate(cpcv_folds):
                for senario in fold.keys():
                    train = fold[senario]["train"]
                    test = fold[senario]["test"]
                    valid = fold[senario]["valid"]

                    train_dataset = pd.DataFrame()
                    for k in train.keys():
                        _df = pd.read_parquet(train[k], engine="pyarrow")
                        train_dataset = train_dataset.append(_df, ignore_index=True)

                    test_dataset = pd.DataFrame()
                    for k in test.keys():
                        _df = pd.read_parquet(test[k], engine="pyarrow")
                        test_dataset = test_dataset.append(_df, ignore_index=True)

                    evaluate_info = {}
                    for k in valid.keys():
                        evaluate_info[k] = {}
                        evaluate_df = pd.read_parquet(valid[k], engine="pyarrow")
                        evaluate_info[k]["data"] = evaluate_df
                        filename_dir = os.path.join(evaluate_downstream_directory, asset_name, str(fold_idx))
                        os.makedirs(filename_dir, exist_ok=True)
                        evaluate_info[k]["filename"] = os.path.join(filename_dir, f"{k}.parquet.gzip")

                    train_and_evaluate(model_type, model, params, train_dataset, test_dataset, evaluate_info)