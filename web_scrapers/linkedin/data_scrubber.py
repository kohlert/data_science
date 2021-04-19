from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class LinkedinScrubber(object):
    def __init__(self, directory='linkedin/data/raw/'):
        self.directory = directory

    def parse_html(self, uid):
        with open(self.directory + uid + '.txt', 'r') as file:
            text = file.read()
        if text:
            soup = BeautifulSoup(text)
            strings = [st for st in soup.stripped_strings]
            return strings

    def build_dataframe(self, uids: list):
        df = pd.DataFrame(columns=['uid', 'strings'])
        for uid in uids:
            st_vector = self.parse_html(uid)
            if st_vector:
                df = df.append(pd.DataFrame(data=[[uid, st_vector]], columns=['uid', 'strings']))
        return df

    def build_matrix(self, uids: list):
        df = self.build_dataframe(uids)
        data_list = []
        for idx, data in df.iterrows():
            st_vector = ' '.join(data['strings'])
            data_list.append(st_vector)
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform(data_list)
        return df['uid'], matrix, vectorizer
