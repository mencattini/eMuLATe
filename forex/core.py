import numpy as np
# to set an defined behavior
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)

import pandas as pd
from functools import reduce
import tqdm
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras import optimizers
import matplotlib.pyplot as plt
import seaborn as sns
from keras import backend as K
from numba import jit

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_csv(filename='/home/romain/gitFile/eMuLATe/arl/data/EURUSD.dat'):
    with open(filename) as f:
        elements = []
        # we split each line in ask and bid
        for ele in f.readlines():
            ask_bid = ele.split(' ')[2]
            splited = ask_bid.split('/')
            ask, bid = splited[0], splited[1]
            elements.append(np.array([np.double(ask), np.double(bid)]))
    res = pd.DataFrame(elements, columns=["ask", "bid"])
    res.to_hdf('EURUSD.hdf', 'eurusd')
    return res


def read_hdf(filename='./EURUSD.hdf'):
    return pd.read_hdf(filename, key='eurusd')


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
    model.add(Dense(N, input_shape=(N,), activation='tanh'))
    model.add(Dense(N, activation='tanh'))
    # the last layers
    model.add(Dense(1))

    # add a sdg
    sgd = optimizers.SGD(lr=0.01, decay=1e-5, momentum=0.75, nesterov=True)

    model.compile(
        loss='mean_absolute_percentage_error', optimizer=sgd,
        metrics=['accuracy']
    )
    return model


def loop(test, test_classes):
        # we apply the sign function because we guess the futur returns
        y_res = np.sign(
            np.array(model.predict_on_batch(test)).transpose()[0]
        )
        res = np.zeros(test.shape[0] - 1)
        i = 0
        # delta * | F_t - F_{t-1} | but vectorized
        cost = np.abs(y_res[1:] - y_res[0:-1]) * 0.0002
        # F_{t-1} * r_t but vecotrized
        for F_t, r_t, c in zip(y_res[0:-1], test_classes, cost):
            # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
            res[i] = F_t * r_t - c
            i += 1
        return res


def evaluation_loop(n, model, df, returns, windowSize):
    # we do somes steps of 2000
    m = 2000
    o = 500
    N = windowSize

    p_t = np.array([])

    # we compile the function with numba
    compiled_loop = jit(nogil=True)(loop)

    for i in tqdm.tqdm(np.arange(m, n, o)):
        # we select the train and test set
        train = np.array(df[i - m:i])
        test = np.array(df[i:i + o])

        # model.fit(
        #     train, returns[i - m + N:i + N], batch_size=o,
        #     verbose=0, epochs=10
        # )
        model.train_on_batch(train, returns[i - m:i])
        # now we compute the cumulative profit
        p_t = np.append(
            p_t,
            compiled_loop(test, returns[i + N: i + o + N - 1])
        )

    return np.cumsum(p_t)


if __name__ == '__main__':

    windowSize = 15
    n = 1000000
    model = init_model(windowSize)
    new = False

    # we get the dataframe, and on it, we compute the returns using
    # pct_change then we take all the dataframe except the first Nan
    prices = None
    if (new):
        prices = pd.DataFrame(read_csv())
    else:
        prices = read_hdf()

    returns = prices.pct_change().fillna(0)
    # update the indexs
    # TODO correct the code under
    returns = returns.set_index(returns.index - 1)

    # the df.iloc[0] is the time 0 to time windowSize - 1
    # the classe of the line is returns[0 + windowSize]
    df = rolling_window(returns['ask'].values, windowSize)

    p_t = evaluation_loop(
        n, model, df, returns['ask'].values, windowSize
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
