import os


class DATAFOLDER:
    # Change your path.
    current_dir = os.getcwd()
    raw_data_root_path = os.path.join(current_dir, "raw_data/g-research-crypto-forecasting/")
    cleaned_data_root_path = os.path.join(current_dir, "data/")


class ASSET_INFO:
    asset_info = {
        "Bitcoin Cash": {"id": 2, "weight": 2.3978952727983707},
        "Binance Coin": {"id": 0, "weight": 4.30406509320417},
        "Bitcoin": {"id": 1, "weight": 6.779921907472252},
        "EOS.IO": {"id": 5, "weight": 1.3862943611198906},
        "Ethereum Classic": {"id": 7, "weight": 2.079441541679836},
        "Ethereum": {"id": 6, "weight": 5.8944028342648505},
        "Litecoin": {"id": 9, "weight": 2.3978952727983707},
    }
