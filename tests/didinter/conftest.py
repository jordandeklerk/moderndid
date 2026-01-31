import pytest

from moderndid import load_favara_imbs


@pytest.fixture
def favara_imbs_data():
    return load_favara_imbs()
