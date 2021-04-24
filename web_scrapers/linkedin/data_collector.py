import time
from inspect import isclass
import pickle
from tqdm import tqdm
import pandas as pd


class DataCollector(object):
    def __init__(self, source_client, directory='linkedin/data/', name=None, pwd=None):
        self.client = source_client() if isclass(source_client) else source_client
        self.client_class = source_client if isclass(source_client) else source_client.__class__
        self.directory = directory
        self.name = name
        self.pwd = pwd

    def collect_with_csv(self, csv_path, columns: list, prefix='', start=0, append=True):
        """
        Uses a csv to feed an iterator to 'collect_data' and a standard set of summary operations from a source client.
        :param csv_path: path of csv configuration file
        :param columns: list of column names from csv to use for iterating over
        :param prefix: string for labeling output files
        :param start: index at which to begin iterating
        :param append: boolean if existing data in csv_path should be appended or overwritten
        :return: a list of dataframes with summary data for each search specified
        """
        searches = pd.read_csv(csv_path)
        searches = searches.loc[start:, columns].values.tolist()
        methods = [self.client_class.oneline_summary, self.client_class.get_summary_data]
        dfs = self.collect_data(methods, searches, prefix=prefix, append=append, start=0)
        return dfs

    def collect_data(self, methods: list, inputs: list, prefix='', start=0, append=True):
        """
        Iterates over each set of inputs in for each set of methods.  Backups are also saved.
        :param methods: list of methods to be called on the data source client
        :param inputs: list of inputs to be passed to the methods being called
        :param prefix: (optional) a string for labeling output files
        :param start: index at which to begin iterating
        :param append: boolean if existing data in csv_path should be appended or overwritten
        :return: a list of dataframes, one for each method
        """
        client = self.client
        client.load_client(name=self.name, pwd=self.pwd)
        inputs = inputs[start:]
        dfs = [getattr(client, method.__name__)(*inputs[0]) for method in methods]
        if append:
            for i in range(len(methods)):
                try:
                    dfs[i] = dfs[i].append(pd.read_csv(self.directory + prefix + methods[i].__name__)).drop_duplicates()
                except FileNotFoundError:
                    continue
        for i in tqdm(range(1, len(inputs))):
            for j in range(len(methods)):
                search = inputs[i]
                method = methods[j]
                df = None
                df = getattr(client, method.__name__)(*search)
                if not (df is None):
                    dfs[j] = dfs[j].append(df).drop_duplicates()
                    dfs[j].to_csv(self.directory + prefix + method.__name__, index=False)
                time.sleep(3)
        return dfs
