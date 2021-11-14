import unittest
import sys
import pandas as pd
import numpy as np
import os
import itertools

sys.path.append("..")
from preprocess.src.extract_data import splitted_data_timestamps
from common.constants import DATAFOLDER, ASSET_INFO


# extract_data.py test
class TestExtractData(unittest.TestCase):
    def test_splitted_data_index(self):
        n_splits_list = [5, 10]
        asset_info = ASSET_INFO.asset_info

        for n_splits_asset_name in list(itertools.product(n_splits_list, asset_info.keys())):
            with self.subTest(n_splits_asset_name=n_splits_asset_name):
                asset_name = n_splits_asset_name[1]
                n_splits = n_splits_asset_name[0]
                asset_id = asset_info[asset_name]["id"]
                _splitted_data_timestamps = splitted_data_timestamps(
                    n_splits=n_splits,
                    asset_name=asset_name,
                    asset_id=asset_id,
                )

                _timestamps = np.empty(0)
                for key in _splitted_data_timestamps["splitted_timestamps"].keys():
                    _timestamps = np.append(_timestamps, _splitted_data_timestamps["splitted_timestamps"][key])

                original_timestamps = pd.read_parquet(
                    os.path.join(DATAFOLDER.cleaned_data_root_path, "Bitcoin.parquet.gzip"),
                    engine="pyarrow",
                )
                target_timestamps = original_timestamps.index.values

                self.assertEqual(asset_name, _splitted_data_timestamps["asset_name"])
                self.assertEqual(asset_id, _splitted_data_timestamps["asset_id"])
                self.assertEqual(n_splits, len(_splitted_data_timestamps["splitted_timestamps"].keys()))
                self.assertEqual(0, (_timestamps - target_timestamps).sum())


if __name__ == "__main__":
    unittest.main()
