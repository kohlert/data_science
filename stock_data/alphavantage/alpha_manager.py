import os
import time
import datetime as dt
import pandas as pd
import numpy as np
from framework.data_portals import DataPortal
from framework.utilities import Singleton


class AlphaClient(object):

    def __init__(self, ticker='IBM', interval='1d', full=True, adjusted=True, demo=False, apikey=None):
        self.apikey = self.get_apikey(apikey)
        self.demo = demo
        self.ticker = ticker
        self.interval = interval
        self.full = full
        self.adjusted = adjusted
        self.last500 = [dt.datetime.now() - dt.timedelta(days=2)] * 500

    @staticmethod
    def get_apikey(apikey=None):
        # Check for APIKey in environment variable
        try:
            if apikey is None:
                apikey = os.environ['ALPHAVANTAGE_API_KEY1']
        except Exception as e:
            raise ValueError('No API key was found.')
        return apikey

    def build_url(self, ticker=None, full=True):
        """
        Args:
            ticker: stock ticker symbol being requested
            interval: time interval ("1min, 5min, 15min, 30min, 60min, 1d, 1w, 1M")
            demo: bool for calling demo data set (free, for testing). Default False.
            full: default True.  False calls only last 100 data points.
            adjusted: default True.  False pulls unadjusted prices.
            api_key: user's unique API key for Alphavantage.  Can be retrieved from environment.

        Returns: a string url which can load stock data from alphavantage.
        """
        ticker = self.ticker if ticker is None else ticker
        # Interval ("1min, 5min, 15min, 30min, 60min, 1d, 1w, 1M")
        fdict = {'1min': ['INTRADAY', '&interval=1min'],
                 '5min': ['INTRADAY', '&interval=5min'],
                 '15min': ['INTRADAY', '&interval=15min'],
                 '30min': ['INTRADAY', '&interval=30min'],
                 '60min': ['INTRADAY', '&interval=60min'],
                 '1d': ['DAILY', ''],
                 '1w': ['WEEKLY', ''],
                 '1M': ['MONTHLY', '']}
        size = 'full' if full else 'compact'
        adj = ''
        if self.adjusted and (self.interval in ['1d', '1w', '1M']):
            adj = '_ADJUSTED'
        if not self.demo:
            return 'https://www.alphavantage.co/query?function=TIME_SERIES_{}{}&symbol={}{}&apikey={}&outputsize=' \
                   '{}&datatype=csv'.format(fdict[self.interval][0], adj, ticker, fdict[self.interval][1],
                                            self.apikey, size)
        else:
            return 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=' \
                   'demo&datatype=csv'

    def download_stock_data(self, ticker=None, full=False, all_outputs=False):
        """
        Args:
            ticker: stock ticker symbol being requested
            interval: time interval ("1min, 5min, 15min, 30min, 60min, 1d, 1w, 1M")
            all_outputs: default False returns only 'close'.  True returns [open, high, low, close, volume]
            demo: default False.  True calls demo data set (free, for testing)
            size: default 'full'.  'compact' pulls last 100 data points.
            adjusted: default True.  False pulls unadjusted prices.
            APIkey: user's unique API key for Alphavantage.  Can be retreived from environment.

        Returns: separate numpy arrays with the following data
        if output = 'all': timestamp, open, high, low, close, volume
        if output = 'close': timestamp, close
        """
        alphaurl = self.build_url(ticker, full=full)
        data = self.throttled_call(alphaurl)
        if all_outputs:
            if self.adjusted:
                timestamp, p_open, p_high, p_low, p_close, p_volume = data['timestamp'], data['open'], data['high'], \
                                                                      data['low'], data['adjusted_close'], \
                                                                      data['volume']
                data = data.loc[:, ['timestamp', 'open', 'high', 'low', 'adjusted_close', 'volume']]
            else:
                timestamp, p_open, p_high, p_low, p_close, p_volume = data['timestamp'], data['open'], data['high'], \
                                                                      data['low'], data['close'], data['volume']
            return np.array(timestamp), np.array(p_open), np.array(p_high), np.array(p_low), np.array(p_close), \
                   np.array(p_volume)
        else:
            if self.adjusted:
                p_close = data['adjusted_close']
            else:
                p_close = data['close']
            return np.array(p_close)

    def throttled_call(self, url):
        self.last500.sort()
        while (dt.datetime.now() - self.last500[-5]).total_seconds() < 60:
            time.sleep(1)
        while (dt.datetime.now() - min(self.last500)).total_seconds() < 86400:
            time.sleep(60)
        out = pd.read_csv(url)
        self.last500.remove(min(self.last500))
        self.last500.append(dt.datetime.now())
        return out

    def update_stock_list(self, list_path='Default', start_point=0):
        # generate file location
        if list_path == 'Default':
            list_path = 'C:/Users/kohle/Documents/Machine Learning/Echo State Networks/Stock_Data/List.csv'
        # load list of stock tickers to be updated
        stock_list = np.loadtxt(list_path, dtype='str')
        # process starting point
        if type(start_point) is str:
            start_point = np.argwhere(stock_list == start_point).item()
        i = 1
        for stock in stock_list[start_point:-1]:
            try:
                self.update_stock_data(stock)
                print('Completed ' + stock + ' ' + str(i) + '/' + str(len(stock_list) - start_point))
            except Exception as ex:
                print(f'{stock} could not be updated.')
                print(ex)
            i += 1

    def update_stock_data(self, ticker=None, filepath=None, location=None):
        """
        Downloads, appends, and saves updates to existing stock data files.
        Args:
            filepath: Stock ticker or filepath for stock data.
            interval: 1min, 5min, 15min, 30min, 60min, 1d, 1w, 1M
            ticker: use only if different from filepath
            APIkey: can be retrieved from environment
            location: default uses config for destination.  'csv' stores files in filepath.
        """
        # adjust shorthand notation for default filepath parameters.
        if not ticker:
            ticker = self.ticker
        if not filepath:
            filepath = f"C:/Users/kohle/Documents/Machine Learning/Echo State Networks/Stock_Data/{ticker}-{self.interval}.csv"
        # Load existing data
        try:
            data = pd.read_csv(filepath)
            if self.interval == '1d' and (
                    pd.Timestamp.now().date() - pd.Timestamp(data.timestamp[0]).date()).days < 80:
                full_hist = False
            else:
                full_hist = True
        except FileNotFoundError or AttributeError:
            full_hist = True
            data = False
        alpha_url = self.build_url(ticker, full=full_hist)
        df = self.throttled_call(alpha_url)
        if df.empty:
            raise FileNotFoundError('No Dataframe returned.')
        if "Error Message" in df.iloc[0, 0]:
            print(df.iloc[0, 0])
        elif "Thank you for using Alpha Vantage!" in df.iloc[0, 0]:
            print(df.iloc[0, 0])
            now = dt.datetime.now()
            for i in filter(lambda x: x < now, self.last500):
                self.last500[self.last500.index(i)] = now
        else:
            if data is not False:
                early_date = pd.Timestamp(df.iloc[-1].timestamp)
                df.append(data[pd.to_datetime(data.timestamp).lt(early_date)]).to_csv(filepath, index=False)
            if location == 'csv':
                df.to_csv(filepath, index=False)
            else:
                portal = DataPortal()
                portal.update(df, 'daily_prices', 'default', uid=ticker, method='append')


if __name__ == '__main__':
    ac = AlphaClient()
    ac.update_stock_list(start_point='ELOX')
