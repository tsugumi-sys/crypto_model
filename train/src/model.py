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
    train_dataset: pd.DataFrame,
    test_dataset: pd.DataFrame,
    evaluate_info,
    target_col: str,
    exclude_col: list = ["buy_cost", "sell_cost"],
):
    train_dataset = train_dataset.drop(exclude_col, axis=1)
    test_dataset = test_dataset.drop(exclude_col, axis=1)
    trained_model = None
    if model_type == "rfc":
        X_train, y_train = train_dataset.drop(target_col, axis=1), train_dataset[target_col]
        trained_model = model.fit(X_train, y_train)

    elif model_type in ["lgbc", "lgbr"]:
        X_train, y_train = train_dataset.drop(target_col, axis=1), train_dataset[target_col]
        X_test, y_test = test_dataset.drop(target_col, axis=1), test_dataset[target_col]

        train_data = lgb.Dataset(X_train, y_train)
        test_data = lgb.Dataset(X_test, y_test, reference=train_data)

        trained_model = model.train(params, train_data, num_boost_round=200, valid_sets=[test_data], callbacks=[early_stopping(stopping_rounds=20)])

    for e_key in evaluate_info.keys():
        evaluate_dataset = evaluate_info[e_key]["data"]
        evaluate_dataset = evaluate_dataset.drop(exclude_col, axis=1)
        preds = trained_model.predict(evaluate_dataset.drop(target_col, axis=1).values)
        evaluate_dataset["predict"] = preds
        file_path = ""
        for s in evaluate_info[e_key]["filename"].split("/"):
            if "parquet.gzip" in s:
                file_path += target_col + s
            else:
                file_path += s + "/"
        evaluate_dataset.to_parquet(file_path, engine="pyarrow", compression="gzip")


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
        coin_name = each_asset["coin_name"]
        cpcv_folds = each_asset["cpcv_folds"]
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

                    filename_dir = os.path.join(evaluate_downstream_directory, coin_name, str(fold_idx))
                    os.makedirs(filename_dir, exist_ok=True)
                    evaluate_info[k]["filename"] = os.path.join(filename_dir, f"{k}.parquet.gzip")

                train_and_evaluate(
                    model_type,
                    model,
                    params,
                    train_dataset,
                    test_dataset,
                    evaluate_info,
                    target_col="y_buy",
                    exclude_col=["buy_cost", "sell_cost", "y_sell"],
                )
                train_and_evaluate(
                    model_type,
                    model,
                    params,
                    train_dataset,
                    test_dataset,
                    evaluate_info,
                    target_col="y_sell",
                    exclude_col=["buy_cost", "sell_cost", "y_buy"],
                )
