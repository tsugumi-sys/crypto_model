import unittest
import os
import sys

sys.path.append("..")
from common.constants import DATAFOLDER


# constants.py test
class TestCONSTANS(unittest.TestCase):
    def test_path(self):
        self.assertTrue(os.path.exists(DATAFOLDER.raw_data_root_path), f"{DATAFOLDER.raw_data_root_path} does not exists.")
        self.assertTrue(os.path.exists(DATAFOLDER.cleaned_data_root_path), f"{DATAFOLDER.cleaned_data_root_path} does not exists.")


if __name__ == "__main__":
    unittest.main()
