from typing import List
import os
from pathlib import Path
import enum


class DATAFOLDER:
    # Change your path.

    # current_dir = os.path.abspath("../")
    path = Path(__file__)
    root_dir = path.parent.absolute()
    root_dir = root_dir.parent.absolute()
    raw_data_root_path = os.path.join(root_dir, "raw_data/")
    cleaned_data_root_path = os.path.join(root_dir, "data/")


class COINNAMES(enum.Enum):
    BTCUSDT = "BTCUSDT"

    @staticmethod
    def coin_names() -> List:
        return [v.value for v in COINNAMES.__members__.values()]


class ASSET_INFO:
    asset_info = {
        "BTCUSDT": {},
    }


class COLUMNS:
    default_columns = [
        "OpenTime",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "CloseTime",
        "QuoteAssetVolume",
        "NumberOfTrades",
        "TakerBuyBaseAssetVolume",
        "TakerBuyQuoteAssetVolume",
        "Ignore",
    ]

    target_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
