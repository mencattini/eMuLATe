import numpy as np
# to set an defined behavior
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)

import pandas as pd
from functools import reduce
import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
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
    model.add(Dense(N, input_shape=(N,), activation='tanh'))
    model.add(Dense(N, activation='tanh'))
    # the last layers
    model.add(Dense(1))

    # add a sdg
    sgd = optimizers.SGD(lr=0.01, nesterov=True)

    model.compile(
        loss='mean_absolute_percentage_error', optimizer=sgd
    )
    return model


def loop_without_trailing_loss(y_res, test_classes, prices):
    res = np.zeros(y_res.shape[0] - 1)
    i = 0
    # delta * | F_t - F_{t-1} | but vectorized
    cost = np.abs(y_res[1:] - y_res[0:-1]) * 0.0002
    # F_{t-1} * r_t but vecotrized
    for F_t, r_t, c in zip(y_res[0:-1], test_classes, cost):
        # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
        res[i] = F_t * r_t - c
        i += 1
    return res


def loop(y_res, test_classes, prices):
        res = np.zeros(y_res.shape[0] - 1)
        i = 0
        # delta * | F_t - F_{t-1} | but vectorized
        cost = np.abs(y_res[1:] - y_res[0:-1]) * 0.0002
        lastPositionPrice = prices[i]
        currentPrice = prices[i + 1]
        lastPosition = y_res[i]
        # F_{t-1} * r_t  - delta * | F_t - F_{t-1}
        for F_t, r_t, c in zip(y_res[0:-1], test_classes, cost):
            # if the F_t is different from the lastPosition it means we need to
            # update the lastPositionprice dans the lastPosition
            if (F_t != lastPosition):
                lastPosition = F_t
                lastPositionPrice = currentPrice
            else:
                # we need to check the difference
                diff = currentPrice - lastPositionPrice
                # if  : diff < 0 and we guess short, diff * -1 > 0
                #     : diff > 0 and we guess long, diff * +1 > 0
                # if diff * F_t < 0, it means we did the wrong choice and we
                # need to controlate our loss
                if (diff * lastPosition > -0.0005):
                    F_t *= -1.0
                if (currentPrice < lastPositionPrice and F_t == -1.0):
                    lastPositionPrice = currentPrice
                if (currentPrice > lastPositionPrice and F_t == +1.0):
                    lastPositionPrice = currentPrice
            # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
            res[i] = F_t * r_t - c
            i += 1
            currentPrice = prices[i + 1]
        return res


def evaluation_loop(n, model, df, returns, windowSize):
    # we do somes steps of 2000
    m = 2000
    o = 500
    N = windowSize

    p_t = np.array([])

    # we compile the function with numba
    compiled_loop = jit('f4[:](f4[:],f4[:],f4[:])', nogil=True)(
        loop_without_trailing_loss
    )

    for i in tqdm.tqdm(np.arange(m, n, o)):
        # we select the train and test set
        train = np.array(df[i - m:i])
        test = np.array(df[i:i + o])

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
                prices.iloc[i - 1: i + o].values)
        )

    return np.cumsum(p_t)


def algorithme(prices, windowSize, n, model):
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

    return p_t


if __name__ == '__main__':

    windowSize = 20
    model = init_model(windowSize)

    # # n = 1000000
    n = 2622129
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
