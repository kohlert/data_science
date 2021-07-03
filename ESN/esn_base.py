import pickle
import os
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
# from sklearn.utils.random import sample_without_replacement
# from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.svm import LinearSVR
# from sklearn.neural_network import MLPClassifier
# from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from framework.data_portals.data_portal import DataPortal
from pyESN import ESN

all_tickers = pd.read_csv('C:\\Users\\kohle\\Documents\\Machine Learning\\Echo State Networks\\Stock_Data\\list.csv')[
    'A'].tolist()
pknum = 0
ticker_range = (pknum * 7000, (pknum + 1) * 7000)
ticker_range = (0, len(all_tickers))
delay_minutes = 0
tasks = ['new', 'continue', 'loop_new']  # choose from: ['new', 'predict_all', 'continue', 'combine', 'loop_new']
# tasks = ['continue']
# tasks = ['combine']
tasks = []
new_env = False  # instruction to keep specified model_env (instead of loading old one, when available)
end_int = -1  # condition to set number of iterations to be run
model_env = {
    'all_tickers': all_tickers,
    'tickers': np.random.choice(all_tickers, 500, replace=False),
    'n_res_list': [30, 30, 30, 30, 30, 30, 50, 80],
    'sparsity_list': [0.5, 0.75, 0.8, 0.85, 0.9, 0.925, 0.95, 0.96],
    'train_len': 4000,
    'drop_len': 200,
    'n_trees': 200,
    'n_comp': 10,
    'vol': False,
    'individual': False,
    # 'model_ui': '0145_SGD',
    'model_ui': '0041_SGD',
    'folder_path': 'models/SGD_hinge_loss'
}


class LinDetrend(object):
    lin_trend = None

    def fit(self, X, y, **fit_params):
        self.lin_trend = np.polyfit(range(len(X)), X, 1)
        return self

    def transform(self, X):
        return X - np.polyval(self.lin_trend, range(len(X))).reshape((1, len(X))).T


def individual_fit_results(tickers, model, prep, net, pca=None, new_fit=True, drop_len=200, train_len=4000,
                           test_len=200, vol=True):
    """
    model is assumed to generate a 1,0 classification to either buy or sell
    """
    gen = portal.iter_get_uids('daily_prices', 'default', tickers)
    df = pd.DataFrame()  # Dataframe with tickers and performance metrics
    df1 = pd.DataFrame()  # array of model coefficients
    df2 = pd.DataFrame()  # array of trading results
    df3 = pd.DataFrame()  # array of buy & hold results
    df4 = pd.DataFrame()  # array of predictions from model
    i = 0
    for data in gen:
        print(i)
        x_train, x_test = np.zeros((0, sum(model_env['n_res_list']) + 1)), \
                          np.zeros((0, sum(model_env['n_res_list']) + 1))
        y_train, y_test, y_cv, y_tcv = [], [], [], []
        w_train, w_test = [], []
        log_vol = np.log10(np.array(data['volume'] + 1).reshape((len(data), 1)))
        log_prices = np.log10(np.array(data['adjusted_close']).reshape((len(data), 1)))
        if len(log_prices) > train_len + test_len:
            prep.fit(log_prices[:train_len])
            log_prices = prep.transform(log_prices)
            if vol:
                prep.fit(log_vol[:train_len])
                log_vol = prep.transform(log_vol)
            else:
                log_vol = np.ones((len(data), 1))
            states = net.get_states(log_vol, log_prices)
            # if pca:
            #     states = pca.transform(states)
            x_train = np.vstack((x_train, states[model_env['drop_len']:train_len]))
            y_train += np.sign((np.sign(
                log_prices[model_env['drop_len'] + 1:train_len + 1, 0] - log_prices[model_env['drop_len']:train_len,
                                                                         0]) + 1) / 2).tolist()
            w_train += np.abs(
                log_prices[model_env['drop_len'] + 1:train_len + 1, 0] - log_prices[model_env['drop_len']:train_len,
                                                                         0]).tolist()
            y_cv += (log_prices[model_env['drop_len'] + 1:train_len + 1, 0] - log_prices[
                                                                              model_env['drop_len']:train_len,
                                                                              0]).tolist()
            x_test = np.vstack((x_test, states[train_len:-1]))
            y_test += np.sign(
                (np.sign(log_prices[train_len + 1:, 0] - log_prices[train_len:-1, 0]) + 1) / 2).tolist()
            w_test += np.abs(log_prices[train_len + 1:, 0] - log_prices[train_len:-1, 0]).tolist()
            y_tcv += (log_prices[train_len + 1:, 0] - log_prices[train_len:-1, 0]).tolist()
            if pca:
                states = pca.transform(states)
                x_train = pca.transform(x_train)
                x_test = pca.transform(x_test)
            if new_fit:
                model.fit(x_train, y_train, sample_weight=w_train)
            acc = model.score(states[1:], np.sign((np.sign(log_prices[1:, 0] - log_prices[:-1, 0]) + 1) / 2).tolist())
            pred = model.predict(states[drop_len:])
            hold = np.array(np.log10(data['adjusted_close'])[drop_len:])
            trading = np.hstack((hold[0], (hold[0] + ((2 * pred[:-1] - 1) * (hold[1:] - hold[:-1])).cumsum())))
            all_hold_ret = hold[-1] - hold[0]
            all_trade_ret = trading[-1] - trading[0]
            all_inc_ret = all_trade_ret / abs(all_hold_ret) - 1
            train_hold_ret = hold[train_len - drop_len] - hold[0]
            train_trade_ret = trading[train_len - drop_len] - trading[0]
            train_inc_ret = train_trade_ret / abs(train_hold_ret) - 1
            test_hold_ret = hold[train_len + test_len - drop_len] - hold[train_len - drop_len]
            test_trade_ret = trading[train_len + test_len - drop_len] - trading[train_len - drop_len]
            test_inc_ret = test_trade_ret - test_hold_ret
            if isinstance(df2, pd.DataFrame):
                df2 = np.pad(trading[:train_len + test_len],
                             [0, train_len + test_len - len(trading[:train_len + test_len])])
                df3 = np.pad(hold[:train_len + test_len],
                             [0, train_len + test_len - len(hold[:train_len + test_len])])
                # df1 = model._get_coef()            #Support Vector Classifier (SVC)
                # df1 = model.feature_importances_   #Random Forest (RF)
                df1 = model.coef_  # SGDClassifier (SGD)
                df4 = np.pad(pred[:train_len + test_len],
                             [0, train_len + test_len - len(pred[:train_len + test_len])])
                df = df.append(pd.DataFrame([[tickers[i], acc, all_hold_ret, all_trade_ret, all_inc_ret,
                                              train_hold_ret, train_trade_ret, train_inc_ret,
                                              test_hold_ret, test_trade_ret, test_inc_ret]],
                                            columns=['ticker', 'accuracy', 'all_hold_ret', 'all_trade_ret',
                                                     'all_inc_ret', 'train_hold_ret', 'train_trade_ret',
                                                     'train_inc_ret', 'test_hold_ret', 'test_trade_ret',
                                                     'test_inc_ret']))
            else:
                df2 = np.vstack((df2, np.pad(trading[:train_len + test_len],
                                             [0, train_len + test_len - len(trading[:train_len + test_len])])))
                df3 = np.vstack((df3, np.pad(hold[:train_len + test_len],
                                             [0, train_len + test_len - len(hold[:train_len + test_len])])))
                df1 = np.vstack((df1, model.coef_))
                # df1 = np.vstack((df1, model._get_coef()))
                # df1 = np.vstack((df1, model.feature_importances_()))
                df4 = np.vstack((df4, np.pad(pred[:train_len + test_len],
                                             [0, train_len + test_len - len(pred[:train_len + test_len])])))
                df = df.append(pd.DataFrame([[tickers[i], acc, all_hold_ret, all_trade_ret, all_inc_ret,
                                              train_hold_ret, train_trade_ret, train_inc_ret,
                                              test_hold_ret, test_trade_ret, test_inc_ret]],
                                            columns=['ticker', 'accuracy', 'all_hold_ret', 'all_trade_ret',
                                                     'all_inc_ret', 'train_hold_ret', 'train_trade_ret',
                                                     'train_inc_ret', 'test_hold_ret', 'test_trade_ret',
                                                     'test_inc_ret']))
        i += 1
    df.reset_index(drop=True, inplace=True)
    return df, df1, df2, df3, df4


def inspect_ticker(ticker, model, prep, net, pca=None, vol=None, drop_len=200):
    data = portal.get('daily_prices', 'default', ticker)
    pp = np.log10(np.array(data['adjusted_close']).reshape((len(data), 1)))
    prep.fit(pp[:model_env['train_len']])
    pp = prep.transform(pp)
    if vol:
        log_vol = np.log10(np.array(ticker['volume'] + 1).reshape((len(ticker), 1)))
        prep.fit(log_vol[:model_env['train_len']])
        log_vol = prep.transform(log_vol)
    else:
        log_vol = np.ones((len(data), 1))
    states = net.get_states(log_vol, pp)
    if pca:
        states = pca.transform(states)
    pred = model.predict(states[drop_len:])
    # score = trading_score()
    hold = data['adjusted_close'][drop_len:]
    trading = hold[0] + ((2 * pred[:-1] - 1) * (hold[1:] - hold[:-1])).cumsum()
    return hold, trading


def plot_ticker(ticker, model, prep, net, pca=None, vol=False):
    hold, trading = inspect_ticker(ticker, model, prep, net, pca=pca, vol=vol)
    plt.plot(hold, label=ticker)
    plt.plot(trading, label=ticker + '_ESN')
    plt.legend()


def generate_plots(tickers, model, prep, net):
    for ticker in tickers:
        print(ticker)
        yield plot_ticker(ticker, model, prep, net)


def trading_score(y, y_pred):
    return sum(y * np.sign(y_pred)) / sum(y * np.sign(y))


def combine_pickles(model_uis, path, keys=('out', 'coefs', 'trading', 'hold', 'pred')):
    """ Combines dictionaries of arrays (saved as separate pickles) into in a single dictionary of arrays """
    data_dict = {}
    if isinstance(model_uis, str):
        model_uis = [model_uis]
    for model_ui in model_uis:
        data_dict[model_ui] = dict(zip(keys, [None] * len(keys)))
        for frame in keys:
            with open(f'{path}/{model_ui}/{model_ui}_{frame}0.pkl', 'rb') as file:
                data_dict[model_ui][frame] = pickle.load(file)
        for frame in keys:
            for i in range(1, pknum + 1):
                with open(f'{path}/{model_ui}/{model_ui}_{frame}{i}.pkl', 'rb') as file:
                    df = pickle.load(file)
                if isinstance(df, pd.DataFrame):
                    data_dict[model_ui][frame] = data_dict[model_ui][frame].append(df)
                else:
                    data_dict[model_ui][frame] = np.vstack((data_dict[model_ui][frame], df))
    return data_dict.copy()


def predict_all(model_env, ticker_range, all_tickers, pknum=0, new_env=True):
    path = model_env["folder_path"]
    with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_model_def.pkl', 'rb') as file:
        model_def = pickle.load(file)
    if not new_env:
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_model_env.pkl', 'rb') as file:
            model_env = pickle.load(file)
    out, coefs, trading, hold, pred = pd.DataFrame(), None, None, None, None
    for batch in range(ticker_range[0], ticker_range[-1], 25):
        df, df1, df2, df3, df4 = individual_fit_results(all_tickers[batch:batch + 25],
                                                        model_def['model'], model_def['prep'],
                                                        model_def['net'], pca=model_def['pca'],
                                                        new_fit=model_env['individual'],
                                                        train_len=model_env['train_len'],
                                                        vol=model_env['vol'], drop_len=model_env['drop_len'])
        out = out.append(df)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_out{pknum}.pkl', 'wb+') as file:
            pickle.dump(out, file)
        if coefs is None:
            coefs, trading, hold, pred = df1, df2, df3, df4
        else:
            trading = np.vstack((trading, df2))
            coefs = np.vstack((coefs, df1))
            hold = np.vstack((hold, df3))
            pred = np.vstack((pred, df4))
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_coefs{pknum}.pkl', 'wb+') as file:
            pickle.dump(coefs, file)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_trading{pknum}.pkl', 'wb+') as file:
            pickle.dump(trading, file)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_hold{pknum}.pkl', 'wb+') as file:
            pickle.dump(hold, file)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_pred{pknum}.pkl', 'wb+') as file:
            pickle.dump(pred, file)


def continue_predict(model_env, ticker_range, all_tickers, pknum=0, new_env=True):
    path = model_env["folder_path"]
    with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_model_def.pkl', 'rb') as file:
        model_def = pickle.load(file)
    if not new_env:
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_model_env.pkl', 'rb') as file:
            model_env = pickle.load(file)
    out, coefs, trading, hold, pred = pd.DataFrame(), None, None, None, None
    for batch in range(ticker_range[0], ticker_range[-1], 25):
        df, df1, df2, df3, df4 = individual_fit_results(all_tickers[batch:batch + 25], model_def['model'],
                                                        model_def['prep'], model_def['net'], pca=model_def['pca'],
                                                        new_fit=model_env['individual'], vol=model_env['vol'],
                                                        train_len=model_env['train_len'],
                                                        drop_len=model_env['drop_len'])
        out = out.append(df)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_out{pknum}.pkl', 'wb+') as file:
            pickle.dump(out, file)
        if coefs is None:
            coefs, trading, hold, pred = df1, df2, df3, df4
        else:
            trading = np.vstack((trading, df2))
            coefs = np.vstack((coefs, df1))
            hold = np.vstack((hold, df3))
            pred = np.vstack((pred, df4))
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_coefs{pknum}.pkl', 'wb+') as file:
            pickle.dump(coefs, file)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_trading{pknum}.pkl', 'wb+') as file:
            pickle.dump(trading, file)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_hold{pknum}.pkl', 'wb+') as file:
            pickle.dump(hold, file)
        with open(f'{path}/{model_env["model_ui"]}/{model_env["model_ui"]}_pred{pknum}.pkl', 'wb+') as file:
            pickle.dump(pred, file)


def collect_model_outputs(path, start=None, end=None):
    data_dict = {}
    model_list = os.listdir(path)
    if start:
        model_list = list(lambda x: x >= start, model_list)
    if end:
        model_list = list(lambda x: x <= end, model_list)
    for model_ui in model_list:
        data_dict[model_ui] = combine_pickles(model_ui, path)
        print(model_ui)
    return data_dict


def get_stats(data_dict, idx, analysis_period):
    """ calculates analysis metrics from dictionaries of outputs """
    df = pd.DataFrame(
        columns=['model', 'ticker', 'idx', 'analysis_period', 'pred_var', 'pred_avg', 'next_trade', 'next_hold'])
    for period in analysis_period:
        for i in idx:
            for key in data_dict.keys():
                tickers = data_dict[key]['out'].ticker
                pred_var = data_dict[key]['pred'][:, i - period:i].var(axis=1)
                pred_avg = data_dict[key]['pred'][:, i - period:i].mean(axis=1)
                next_trade = (data_dict[key]['trading'][:, i + 1]) - (data_dict[key]['trading'][:, i])
                next_hold = (data_dict[key]['hold'][:, i + 1]) - (data_dict[key]['hold'][:, i])
                try:
                    df1 = pd.DataFrame(columns=['ticker', 'pred_var', 'pred_avg', 'next_trade', 'next_hold'],
                                       data=np.vstack(([tickers], [pred_var], [pred_avg], [next_trade], [next_hold])).T)
                    df1['model'] = key
                    df1['idx'] = i
                    df1['analysis_period'] = period
                    df = df.append(df1)
                except ValueError as ex:
                    print(ex)
                    print(key)
            print((period, i))
    return df


def compile_stats(path):
    """ combines all items from the given folder of stats arrays """
    df = pd.DataFrame()
    for item in os.listdir(path):
        print(item)
        with open(path + '/' + item, 'rb') as file:
            df1 = pickle.load(file)
            # df1 = df1.loc[df1.pred_var < 1.0]
            # df1 = df1.loc[df1.pred_var > 0.0]
            df1 = df1.loc[df1.next_hold != np.inf]
            df1 = df1.loc[df1.next_hold != -np.inf]
            df = df.append(df1)
    return df


def calculate(model_ui, path):
    data = combine_pickles(model_ui, path)
    df = get_stats(data, range(3999, 4199, 5), [2, 3, 4, 5, 7, 10, 15, 25, 50, 75, 100, 150, 200])
    with open(f'{path}_stats/{model_ui[:4]}_stats.pkl', 'wb+') as file:
        pickle.dump(df, file)


# items = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(calculate)(ui, model_env['folder_path']) for ui in os.listdir('models/SGD_hinge_loss'))

if __name__ == '__main__':
    import time
    import datetime

    print(datetime.datetime.today())
    print(f'pickle number: {str(pknum)}')
    print(f'delay: {delay_minutes} min')
    print(f'model_ui: {model_env["model_ui"]}')
    print(f'tasks: {str(tasks)}')
    time.sleep(delay_minutes * 60)
    portal = DataPortal()
    loop = 0
    while loop != end_int:
        if 'new' in tasks:
            gen = portal.iter_get_uids('daily_prices', 'default', model_env['tickers'])
            net = ESN(1, 1, n_reservoir=model_env['n_res_list'][0], sparsity=model_env['sparsity_list'][0], noise=0)
            for i in range(1, len(model_env['n_res_list'])):
                temp_net = ESN(1, 1, n_reservoir=model_env['n_res_list'][i], sparsity=model_env['sparsity_list'][i],
                               noise=0)
                net.merge(temp_net)
            x_train, x_test = np.zeros((0, sum(model_env['n_res_list']) + 1)), np.zeros(
                (0, sum(model_env['n_res_list']) + 1))
            y_train, y_test, y_cv, y_tcv = [], [], [], []
            w_train, w_test = [], []
            prep = Pipeline([('detrend', LinDetrend()), ('scaler', StandardScaler())])
            for ticker in gen:
                log_prices = np.log10(np.array(ticker['adjusted_close']).reshape((len(ticker), 1)))
                if len(log_prices) > model_env['train_len']:
                    prep.fit(log_prices[:model_env['train_len']])
                    log_prices = prep.transform(log_prices)
                    if model_env['vol']:
                        log_vol = np.log10(np.array(ticker['volume'] + 1).reshape((len(ticker), 1)))
                        prep.fit(log_vol[:model_env['train_len']])
                        log_vol = prep.transform(log_vol)
                    else:
                        log_vol = np.ones((len(ticker), 1))

                    states = net.get_states(log_vol, log_prices)
                    x_train = np.vstack((x_train, states[model_env['drop_len']:model_env['train_len']]))
                    y_train += np.sign((np.sign(
                        log_prices[model_env['drop_len'] + 1:model_env['train_len'] + 1, 0] - log_prices[
                                                                                              model_env['drop_len']:
                                                                                              model_env['train_len'],
                                                                                              0]) + 1) / 2).tolist()
                    w_train += np.abs(log_prices[model_env['drop_len'] + 1:model_env['train_len'] + 1, 0] - log_prices[
                                                                                                            model_env[
                                                                                                                'drop_len']:
                                                                                                            model_env[
                                                                                                                'train_len'],
                                                                                                            0]).tolist()
                    y_cv += (log_prices[model_env['drop_len'] + 1:model_env['train_len'] + 1, 0] - log_prices[model_env[
                                                                                                                  'drop_len']:
                                                                                                              model_env[
                                                                                                                  'train_len'],
                                                                                                   0]).tolist()
                    x_test = np.vstack((x_test, states[model_env['train_len']:-1]))
                    y_test += np.sign((np.sign(
                        log_prices[model_env['train_len'] + 1:, 0] - log_prices[model_env['train_len']:-1,
                                                                     0]) + 1) / 2).tolist()
                    w_test += np.abs(
                        log_prices[model_env['train_len'] + 1:, 0] - log_prices[model_env['train_len']:-1, 0]).tolist()
                    y_tcv += (log_prices[model_env['train_len'] + 1:, 0] - log_prices[model_env['train_len']:-1,
                                                                           0]).tolist()

            # score_parameters = {'score_weights': w_train}
            weighted_scorer = make_scorer(trading_score, greater_is_better=True)
            # ToDo: add random scrambling to training data

            # model = RandomForestClassifier(verbose=True, n_estimators=model_env['n_trees'], n_jobs=-1, criterion='entropy', max_features='sqrt', max_depth=8)
            # model = RandomForestRegressor(n_estimators=2, verbose=10, n_jobs=8, criterion='mae', max_features='sqrt', max_depth=8)
            # model = LinearSVR(max_iter=10000)
            # model = MLPClassifier(verbose=True, hidden_layer_sizes=(50, 10, 30, 4), learning_rate='adaptive', max_iter=500,
            #                      warm_start=True, tol=0.00000001, n_iter_no_change=20, activation='tanh')
            # model = SVC(cache_size=3000, verbose=10, kernel='linear')
            model = SGDClassifier(verbose=10, n_jobs=-1)
            pca = PCA(n_components=model_env['n_comp'])
            pc_x_train = pca.fit_transform(x_train)
            pc_x_test = pca.transform(x_test)
            model.fit(pc_x_train, y_train, sample_weight=w_train)
            model_def = {'model': model,
                         'net': net,
                         'prep': prep,
                         'pca': pca,
                         }
            if not os.path.isdir(f'{model_env["folder_path"]}/{model_env["model_ui"]}'):
                os.mkdir(f'{model_env["folder_path"]}/{model_env["model_ui"]}')
            with open(f'{model_env["folder_path"]}/{model_env["model_ui"]}/{model_env["model_ui"]}_model_env.pkl',
                      'wb+') as file:
                pickle.dump(model_env, file)
            with open(f'{model_env["folder_path"]}/{model_env["model_ui"]}/{model_env["model_ui"]}_model_def.pkl',
                      'wb+') as file:
                pickle.dump(model_def, file)
            # model.fit(x_train, y_cv)
            # try:
            #     selector = RFECV(model, step=0.33, cv=5, scoring=weighted_scorer)
            #     selector.fit(x_train, y_cv)
            #     df = individual_results(all_tickers, selector, prep, net)
            # except Exception as ex:
            #     print(ex)
            #     df = individual_results(all_tickers, model, prep, net)
            # df = individual_results(all_tickers, model, prep, net)
        if 'predict_all' in tasks:
            predict_all(model_env, ticker_range, all_tickers, pknum=pknum, new_env=new_env)
        if 'continue' in tasks:
            continue_predict(model_env, ticker_range, all_tickers, pknum=pknum, new_env=new_env)
        if 'combine' in tasks:
            data_dict = combine_pickles(model_env['model_ui'], model_env['folder_path'])
        if 'loop_new' in tasks:
            loop += 1
            num = int(model_env['model_ui'][:4]) + 1
            mod = model_env['model_ui'][4:]
            model_env['model_ui'] = f'{num :04d}' + mod
            model_env['tickers'] = np.random.choice(all_tickers, 500, replace=False)
        else:
            break
        if 'sandbox' in tasks:
            df = compile_stats('models/SGD_hinge_loss_stats')
            df['next_ideal'] = np.sign(df.next_hold).astype(int)
            df.reset_index(inplace=True, drop=True)
            train_idx = np.random.choice(df.index, 10000000, replace=False)
            train_idx.sort()
            df['pred'] = (np.sign(df.next_trade) / np.sign(df.next_hold.replace({0.0: 1.0}))).astype(int)
            df_train = df.loc[train_idx, ['model', 'ticker', 'analysis_period', 'pred_var', 'pred_avg', 'pred']]
            df_test = df.loc[:, ['model', 'ticker', 'analysis_period', 'pred_var', 'pred_avg', 'pred']].drop(train_idx)
            y_train = df.loc[train_idx, 'next_ideal']
            y_test = df.loc[:, 'next_ideal'].drop(train_idx)
            df_train = df_train.astype(dict(zip(['model', 'ticker', 'analysis_period', 'pred_var', 'pred_avg', 'pred'],
                                                [str, str, int, float, float, int])))
            df_test = df_test.astype(dict(zip(['model', 'ticker', 'analysis_period', 'pred_var', 'pred_avg', 'pred'],
                                              [str, str, int, float, float, int])))
            from sklearn.preprocessing import OneHotEncoder
            from sklearn.experimental import enable_hist_gradient_boosting
            from sklearn.ensemble import HistGradientBoostingClassifier
            from sklearn.compose import make_column_transformer
            from sklearn.compose import make_column_selector
            from sklearn.pipeline import make_pipeline

            one_hot_encoder = make_column_transformer(
                (OneHotEncoder(sparse=False, handle_unknown='ignore'),
                 make_column_selector(dtype_include='object')),
                remainder='passthrough')

            gbc_one_hot = make_pipeline(one_hot_encoder,
                                        HistGradientBoostingClassifier(verbose=10))
            gbc_one_hot.fit(df_train.loc[:, ['model', 'analysis_period', 'pred_var', 'pred_avg', 'pred']], y_train)

            df['next_ideal'] = np.sign(df.next_hold).astype(int)
            df.reset_index(inplace=True, drop=True)
            # train_idx = np.random.choice(df.index, 8000000, replace=False)
            train_idx = df.loc[df.idx < 4120].index.tolist()
            train_idx.sort()
            df['pred'] = (np.sign(df.next_trade) / np.sign(df.next_hold.replace({0.0: 1.0}))).astype(int)
            df_train = df.loc[train_idx, ['model', 'pred_var', 'pred_avg', 'pred']]
            df_test = df.loc[:, ['model', 'pred_var', 'pred_avg', 'pred']].drop(train_idx)
            y_train = df.loc[train_idx, 'next_ideal']
            y_test = df.loc[:, 'next_ideal'].drop(train_idx)
            df_train = df_train.astype(dict(zip(['model', 'pred_var', 'pred_avg', 'pred'], [str, float, float, int])))
            df_test = df_test.astype(dict(zip(['model', 'pred_var', 'pred_avg', 'pred'], [str, float, float, int])))
            df_train.pred_avg = df_train.pred_avg.astype(float)
            df_train.pred_var = df_train.pred_var.astype(float)
            df_test.pred_avg = df_test.pred_avg.astype(float)
            df_test.pred_var = df_test.pred_var.astype(float)
            gbc_one_hot.fit(df_train, y_train)
            p = gbc_one_hot.predict(df_test)
            act = df.loc[:, 'next_hold'].drop(train_idx)
            res = p * act
            p_t = gbc_one_hot.predict(df_train)
            act_t = df.loc[train_idx, 'next_hold']
            res_t = p_t * act_t
