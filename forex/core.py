
from utils import read_csv, read_hdf, plot_all
from Arl import Arl
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':

    windowSize = 20
    arl = Arl(20)

    n = 1000000
    # n = 2622129
    new = False

    # we get the dataframe, and on it, we compute the returns using
    # pct_change then we take all the dataframe except the first Nan
    prices = None
    if (new):
        prices = pd.DataFrame(read_csv(n=n))
    else:
        prices = read_hdf(n=n)
    title_two = "EUR/USD"

    # prices = pd.read_csv('./data/EURCHF.csv', delimiter=';', header=None)
    # prices = pd.DataFrame(prices[1].values)
    # prices.columns = ["ask"]
    # n = prices.shape[0]
    # title_two = "EUR/CHF"

    p_t = arl.run(prices, n)
    plot_all(p_t, prices, n, title_two=title_two)
