import pandas as pd
from arctic.date import DateRange
from datetime import datetime as dt


def iterate_stocks(lib, snapshot=None, date_range=None):
    for sym in lib.list_symbols(snapshot=snapshot):
        yield lib.read(sym, as_of=snapshot, date_range=date_range)


def loop_return_computation(lib, snapshot='alpha'):
    for stock in lib.list_symbols(snapshot=snapshot):
        s = lib.read(stock, date_range=DateRange(dt(1999, 12, 1), dt(2016, 3, 8)), as_of='alpha')
        df_returns = s.data['Adj. Close'].pct_change()
        s.data['Returns'] = df_returns
        lib.write(stock, s.data)


def compute_smart_corr(returns, stock, window=50, min_periods=30):
    """
    Same as compute_corr, but avoiding the computation of the correlation of every column that appears before
    the :stock

    :param returns: a pandas DataFrame containing the returns, where the index is called Date and the columns are the
    different stocks
    :param stock: the name of the column against which the correlations should be computed
    :param window: parameter for rolling_corr
    :param min_periods: parameter for rolling_corr
    :return: pandas DataFrame, with the correlations against each stock in each column
    """
    column_index = returns.columns.tolist().index(stock)
    reduced_returns = returns.iloc[:, column_index:]
    x = pd.rolling_corr(reduced_returns, reduced_returns.loc[:, stock], window=window, min_periods=min_periods)
    assert isinstance(x, pd.DataFrame)
    return prepare_as_panel(x, stock)


def compute_corr(returns, stock, window=50, min_periods=30):
    """
    Computes the rolling correlation of a DataFrame against of one its elements

    :param returns: a pandas DataFrame containing the returns, where the index is called Date and the columns are the
    different stocks
    :param stock: the name of the column against which the correlations should be computed
    :param window: parameter for rolling_corr
    :param min_periods: parameter for rolling_corr
    :return: pandas DataFrame, with the correlations against each stock in each column
    """
    x = pd.rolling_corr(returns, returns[stock], window=window, min_periods=min_periods)
    assert isinstance(x, pd.DataFrame)
    return prepare_as_panel(x, stock)


def prepare_as_panel(df, stock_name):
    """
    Auxiliary function for converting a dataframe with correlation values into a DataPanel
    :param df:
    :param stock_name:
    :return:
    """
    df['stock'] = stock_name
    return df.reset_index().set_index(['stock', 'Date']).to_panel()


def grouper(n, iterable, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    from itertools import zip_longest
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


def iterate_over_returns(rets, number_items, window_size):
    for x in grouper(number_items, range(window_size,300)):
        yield rets.ix[min(x)-window_size:max(x)+1]


def correlation_panel_as_frame(panel):
    df = panel.swapaxes('major', 'items').to_frame(filter_observations=False)
    return df


def get_last_from(df, n):
    return df.loc[(slice(df.index.levels[0][-n], None), slice(None)), :]


def iterative_correlation_computation(returns_df, chunk_size, window_size):
    for r in iterate_over_returns(returns_df, chunk_size, window_size):
        c = pd.rolling_corr(r, window=window_size, min_periods=window_size-10, pairwise=True)
        df = c.swapaxes('major', 'items').to_frame(filter_observations=False)
        yield get_last_from(df, chunk_size)
