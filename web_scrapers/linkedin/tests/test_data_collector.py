from unittest import TestCase
from ..data_collector import DataCollector
from ..linkedin_public import LinkedinPublic
from ..linkedin_login import LinkedinLogin
import pandas as pd
import os
import json


class TestDataCollector(TestCase):
    client = LinkedinPublic
    name, pwd = None, None
    if client.login_required:
        with open(os.environ['cred'] + 'linkedin.json', 'r') as doc:
            name, pwd = json.load(doc)['linkedin'].values()
    collector = DataCollector(client, name=name, pwd=pwd)

    def test_collect_data(self):
        methods = [LinkedinPublic.get_detailed_data]
        df = pd.read_csv('linkedin/data/DataScienceLead_smax_get_summary_data')
        searches = [[str(id)] for id in df.loc[:, 'uid']]
        dfs = self.collector.collect_data(methods, searches, prefix='DataScienceLead_smax_', start=0)
        assert all(isinstance(dfs[i], pd.DataFrame) for i in range(len(methods)))

    def test_collect_with_csv(self):
        path = 'linkedin/data/locations.csv'
        columns = ['data science lead', 'City', 'State']
        prefix = 'DataScienceLead_all_'
        dfs = self.collector.collect_with_csv(path, columns, prefix=prefix, start=0)
        assert all(isinstance(dfs[i], pd.DataFrame) for i in range(len(dfs)))
