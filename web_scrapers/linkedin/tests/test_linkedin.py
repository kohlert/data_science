from unittest import TestCase
from ...linkedin import LinkedinLogin
from ...linkedin import LinkedinPublic
import pandas as pd
import os
import json


class TestSourceClient(TestCase):
    keyword = 'data science'
    location = 'United States'
    source_client = LinkedinPublic
    # source_client = LinkedinLogin
    uid = '2417292529'
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
        # if self.login_required:
        #     page.load_client(name=self.name, password=self.pwd)
        # else:
        #     page.load_client()
        assert page.driver.session_id

    def test_get_summary_data(self):
        page = self.source_client()
        page.load_client()
        data = page.get_summary_data(self.keyword, self.location)
        # this just sets up the next test
        self.uid = data.iloc[1]['uid']
        assert isinstance(data, pd.DataFrame) and not data['uid'].empty

    def test_get_detailed_data(self):
        page = self.source_client()
        page.load_client()
        data = page.get_detailed_data(self.uid)
        assert isinstance(data, pd.DataFrame)

# class Testlinkedin(TestCase):
#     with open(os.environ['cred'] + 'linkedin.json', 'r') as doc:
#         uid, pwd = json.load(doc)['linkedin'].values()
#     keyword = 'data science'
#     location = 'United States'
#
#     def test_login_linkedin(self):
#         page = Linkedin(self.uid, self.pwd, login=True)
#         assert page.driver.session_id
#
#     def test_job_search(self):
#         page = Linkedin(self.uid, self.pwd, login=True)
#         page.job_search(self.keyword, self.location)
#         assert True
#
#     def test_jobs_summary(self):
#         page = Linkedin(self.uid, self.pwd, login=True)
#         page.job_search(self.keyword, self.location)
#         df = page.jobs_summary()
#         assert isinstance(df, pd.DataFrame)
#
#     def test_collect_all_summary_data(self):
#         page = Linkedin(self.uid, self.pwd, login=True)
#         df = page.collect_all_summary_data(self.keyword, self.location,
#                                            save_path='.\\linkedin\\data\\data_science_USA.csv')
#         assert isinstance(df, pd.DataFrame)
