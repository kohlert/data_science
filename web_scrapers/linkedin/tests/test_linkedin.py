from unittest import TestCase
from ..linkedin_public import LinkedinPublic
from ..linkedin_login import LinkedinLogin
import pandas as pd
import os
import json


class TestSourceClient(TestCase):
    keyword = 'data science'
    location = 'Bakersfield'
    # source_client = LinkedinPublic
    source_client = LinkedinLogin
    uid = '2403641965'
    login_required = source_client.login_required
    name, pwd = None, None
    if login_required:
        with open(os.environ['cred'] + 'linkedin.json', 'r') as doc:
            name, pwd = json.load(doc)['linkedin'].values()

    def test_init(self):
        page = self.source_client()
        assert isinstance(page, self.source_client)

    def test_load_client(self):
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        assert page.driver.session_id

    def test_get_summary_data(self):
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        data = page.get_summary_data(self.keyword, self.location)
        # this just sets up the next test
        self.uid = data.iloc[1]['uid']
        assert isinstance(data, pd.DataFrame) and not data['uid'].empty

    def test_get_detailed_data(self):
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        data = page.get_detailed_data(self.uid)
        assert isinstance(data, pd.DataFrame)

    def test_online_summary(self):
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        page.job_search(self.keyword, self.location)
        df = page.oneline_summary()
        assert isinstance(df, pd.DataFrame) and not df.empty
