import os

import polars as pl
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "favara_imbs_did_multiplegt_dyn.csv.gz")


@pytest.fixture
def favara_imbs_data():
    return pl.read_csv(DATA_PATH)
