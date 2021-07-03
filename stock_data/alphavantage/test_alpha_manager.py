import pandas as pd
from .alpha_manager import AlphaClient


def test_update_stock_data():
    ac = AlphaClient()
    test = ac.update_stock_data()
    assert isinstance(test, pd.DataFrame)


def test_update_stock_list():
    ac = AlphaClient()
    ac.update_stock_list()
