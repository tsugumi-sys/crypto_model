import os
from pathlib import Path


class DATAFOLDER:
    # Change your path.

    # current_dir = os.path.abspath("../")
    path = Path(__file__)
    root_dir = path.parent.absolute()
    root_dir = root_dir.parent.absolute()
    raw_data_root_path = os.path.join(root_dir, "raw_data/")
    cleaned_data_root_path = os.path.join(root_dir, "data/")


class ASSET_INFO:
    asset_info = {
        "BTCUSDT": {},
    }
