import numpy as np
# to set an defined behavior
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)

import pandas as pd
import tqdm
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from numba import jit
from utils import read_csv, read_hdf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# magic
def rolling_window(a, window):
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    windows = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return pd.DataFrame(np.squeeze(windows))


def init_model(N):
    # multiple core
    config = tf.ConfigProto(
        intra_op_parallelism_threads=16,
        inter_op_parallelism_threads=16,
        allow_soft_placement=True
    )
    session = tf.Session(config=config)
    K.set_session(session)
    # the sequential mode
    model = Sequential()

    # without the reccurent layer
    model.add(Dense(N * 5, input_shape=(N,), activation='tanh'))
    # hidden layer
    model.add(Dense(N, activation='relu'))
    # the last layer
    model.add(Dense(1))

    model.compile(
        loss='mean_absolute_percentage_error', optimizer='rmsprop'
    )
    return model


def loop(y_res, test_classes, prices, pnl):
    res = np.zeros(y_res.shape[0] - 2)
    i = 0
    rate = 0.005

    current = pnl
    max_pnl = pnl
    last_position = y_res[i]
    # it's the layer one who only apply the signal
    for F_t, r_t in zip(y_res[1:], test_classes[1:]):

        if F_t != last_position:
            max_pnl = current
            last_position = F_t
        else:
            diff = max_pnl - current
            if diff < 0:
                max_pnl = current
            elif diff > 0 and diff > rate:
                F_t = 0
                y_res[i + 1] = 0

        # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
        res[i] = current * F_t * r_t - 0.0002 * np.abs(F_t - y_res[i])
        res[i] += current
        current = res[i]
        i += 1

    return res


def simple_loop(y_res, test_classes, prices, pnl):
    res = np.zeros(y_res.shape[0] - 2)
    i = 0

    current = pnl
    for F_t, r_t in zip(y_res[1:], test_classes[1:]):
        # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
        res[i] = current * F_t * r_t - 0.0002 * np.abs(F_t - y_res[i])
        res[i] += current
        current = res[i]
        i += 1

    return res


def evaluation_loop(n, model, df, returns, windowSize, prices):
    # we do somes steps of 2000
    m = 2000
    o = 500
    N = windowSize

    p_t = np.array([1])

    # we compile the function with numba
    compiled_loop = jit('f4[:](f4[:],f4[:],f4[:],f4)', nogil=True)(
        loop
    )

    for i in tqdm.tqdm(np.arange(m, n, o)):

        # first part : we treat the data
        # we select the train and test set
        train = np.array(df[i - m:i])
        test = np.array(df[i:i + o])

        # we train our model
        model.train_on_batch(train, returns[i - m:i])
        # we compute the classes
        y_res = np.sign(
            np.array(model.predict_on_batch(test)).transpose()[0]
        )
        # now we compute the cumulative profit
        p_t = np.append(
            p_t,
            compiled_loop(
                y_res, returns[i + N: i + o + N - 1],
                prices.iloc[i - 1: i + o].values, p_t[-1])
        )

    # return np.cumsum(p_t)
    return p_t


def algorithme(prices, windowSize, n, model):
    returns = prices.pct_change().fillna(0)
    # update the indexs
    # TODO correct the code under
    returns = returns.set_index(returns.index - 1)

    # the df.iloc[0] is the time 0 to time windowSize - 1
    # the classe of the line is returns[0 + windowSize]
    df = rolling_window(returns['ask'].values, windowSize)

    p_t = evaluation_loop(
        n, model, df, returns['ask'].values, windowSize, prices['ask']
    )

    sns.set_style("darkgrid")

    plt.subplot(2, 1, 1)
    plt.plot(p_t)
    plt.grid(True)
    plt.title("$Cumulated_{profit}$")
    plt.xlabel("ticks")
    plt.ylabel("profit")
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(prices[0:n])
    plt.grid(True)
    plt.title("Change EUR/USD")
    plt.xlabel("ticks")
    plt.ylabel("prices")
    plt.tight_layout()
    plt.show()

    return p_t


if __name__ == '__main__':

    windowSize = 20
    model = init_model(windowSize)

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

    # prices = pd.read_csv('./data/EURCHF.csv', delimiter=';', header=None)
    # prices = pd.DataFrame(prices[1].values)
    # prices.columns = ["ask"]
    # n = prices.shape[0]

    p_t = algorithme(prices, windowSize, n, model)
