from bs4 import BeautifulSoup
import sklearn as skl
import pandas as pd


class LinkedinScrubber(object):
    def __init__(self, directory='linkedin/data/raw/'):
        self.directory = directory

    def parse_html(self, uid):
        with open(self.directory + uid + '.txt', 'r') as file:
            text = file.read()
        if text:
            soup = BeautifulSoup(text)
            return [uid, [st for st in soup.stripped_strings]]

    def build_dataframe(self, uids: list):
        df = pd.DataFrame(columns=['uid', 'strings'])
        for uid in uids:
            st_vector = self.parse_html(uid)
            if st_vector:
                df = df.append(pd.DataFrame(data=[st_vector], columns=['uid', 'strings']))
        return df
