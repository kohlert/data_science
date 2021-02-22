import time
import pickle
import pandas as pd


class DataCollector(object):
    def __init__(self, source_client, directory='linkedin/data/'):
        self.client = source_client
        self.directory = directory

    def collect_data(self, methods: list, iterators: list):
        """
        Iterates over each set of inputs in 'iterators' for each method in 'methods'.  Backups are also saved.
        :param methods: list of methods to be called on the data source client
        :param iterators: list of inputs to be passed to the methods being called
        :return: a list of dataframes, one for each method
        """
        client = self.client()
        client.load_client()
        dfs = [getattr(client, method.__name__)(*iterators[0]) for method in methods]
        for i in range(1, len(iterators)):
            for j in range(len(methods)):
                inputs = iterators[i]
                method = methods[j]
                df = getattr(client, method.__name__)(*inputs)
                dfs[j] = dfs[j].append(df).drop_duplicates()
                dfs[j].to_csv(self.directory + method.__name__)
                time.sleep(client.sec_per_page)
        return dfs
