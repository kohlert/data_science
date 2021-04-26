from unittest import TestCase
from ..data_collector import DataCollector
from ..linkedin_public import LinkedinPublic
from ..linkedin_login import LinkedinLogin
import pandas as pd
import os
import json


class TestDataCollector(TestCase):
    client = LinkedinPublic(reload=False, sal_bin=4)
    name, pwd = None, None
    if client.login_required:
        with open(os.environ['cred'] + 'linkedin.json', 'r') as doc:
            name, pwd = json.load(doc)['linkedin'].values()
    collector = DataCollector(client, name=name, pwd=pwd)

    def test_collect_data(self):
        methods = [LinkedinPublic.get_detailed_data]
        df = pd.read_csv('linkedin/data/unique_datasciencelead_combined_list.csv')
        searches = [[str(id)] for id in list(set(df.loc[:, 'uid']))]
        dfs = self.collector.collect_data(methods, searches, prefix='master_', start=(2446 + 2980 + 42600))
        assert all(isinstance(dfs[i], pd.DataFrame) for i in range(len(methods)))

    def test_collect_with_csv(self):
        path = 'linkedin/data/locations.csv'
        columns = ['data science', 'City', 'State']
        prefix = 'new_'
        dfs = self.collector.collect_with_csv(path, columns, prefix=prefix, start=0)
        assert all(isinstance(dfs[i], pd.DataFrame) for i in range(len(dfs)))
