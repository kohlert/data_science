from unittest import TestCase
from ..data_collector import DataCollector
from ..linkedin_public import LinkedinPublic
from ..linkedin_login import LinkedinLogin
import pandas as pd


class TestDataCollector(TestCase):
    collector = DataCollector(LinkedinPublic)

    def test_collect_data(self):
        methods = [LinkedinLogin.oneline_summary, LinkedinLogin.get_summary_data]
        searches = [['data science', 'Bakersfield, CA'],
                    ['data science', 'Traverse City, MI'],
                    ['data science', 'Denver']]
        dfs = self.collector.collect_data(methods, searches)
        assert all(isinstance(dfs[i], pd.DataFrame) for i in range(len(methods)))
