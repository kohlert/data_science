from unittest import TestCase
from ..data_scrubber import LinkedinScrubber
import pandas as pd
import os


class TestLinkedinScrubber(TestCase):
    directory = 'linkedin/data/raw/'

    def test_parse_html(self):
        uid = '1653447286'
        test = LinkedinScrubber().parse_html(uid)
        assert isinstance(test, list) and len(test) == 2

    def test_build_dataframe(self):
        uids = ['1653447286', '1549635799', '1509059247']
        test = LinkedinScrubber().build_dataframe(uids)
        test.to_csv('linkedin/data/parsed_details.csv')
        assert isinstance(test, pd.DataFrame) and test.size == 6

    def test_big_dataframe(self):
        uids = [file[:-4] for file in os.listdir(self.directory)]
        test = LinkedinScrubber().build_dataframe(uids)
        assert isinstance(test, pd.DataFrame)
