from unittest import TestCase
from ..linkedin_public import LinkedinPublic
from ..linkedin_login import LinkedinLogin
import pandas as pd
import os
import json


class TestSourceClient(TestCase):
    keyword = 'data science lead'
    state = 'Texas'
    city = 'Austin'
    source_client = LinkedinPublic
    # source_client = LinkedinLogin
    uid = '2424284044'
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
        page.quit_client()

    def test_get_summary_data(self):
        """
        Gets all summary data for results of a specific source_client search
        """
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        data = page.get_summary_data(self.keyword, self.city, self.state)
        assert isinstance(data, pd.DataFrame) and not data['uid'].empty
        page.quit_client()

    def test_get_detailed_data(self):
        """
        Gets details for a specific uid
        """
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        data = page.get_detailed_data(self.uid)
        assert isinstance(data, pd.DataFrame)
        page.quit_client()

    def test_oneline_summary(self):
        """
        Gets oneline summary of search results.
        """
        page = self.source_client()
        page.load_client(name=self.name, pwd=self.pwd)
        df = page.oneline_summary(self.keyword, self.city, self.state)
        assert isinstance(df, pd.DataFrame) and not df.empty
        page.quit_client()
