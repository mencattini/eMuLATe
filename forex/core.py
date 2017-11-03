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


def loop_without_stop_loss(y_res, test_classes, prices):
    res = np.zeros(y_res.shape[0] - 1)
    i = 0

    # F_{t-1} * r_t but vecotrized
    for F_t, r_t in zip(y_res[1:], test_classes[1:]):
        # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
        res[i] = F_t * r_t - 0.0002 * np.abs(F_t - y_res[i])
        i += 1
    return res


def loop(y_res, returns, prices):
        res = np.zeros(y_res.shape[0] - 1)
        i = 0

        maxp_t = res[0]
        lastPosition = y_res[0]
        acc_p_t = res[0]
        # F_{t-1} * r_t  - delta * | F_t - F_{t-1}
        for j in range(1, len(y_res)):
            F_t = y_res[j]
            r_t = returns[j]
            # if the position are different we need to update the max cumulated
            # profit
            if (F_t != lastPosition):
                maxp_t = acc_p_t
                lastPosition = F_t
            else:
                # else if the max cumulated is less than the new cumulated
                # profit we need to update too
                if (maxp_t <= acc_p_t):
                    maxp_t = acc_p_t
                # else we need to control our loss
                else:
                    diff = np.abs(maxp_t) - np.abs(acc_p_t)
                    # diff = maxp_t - acc_p_t
                    if (diff > 0.005):
                        F_t = 0
                        y_res[j] = 0
            # F_{t-1} * r_t - delta * | F_t - F_{t-1} |
            res[i] = F_t * r_t - 0.0002 * np.abs(F_t - y_res[i])
            acc_p_t += res[i]
            i += 1
        return res


def evaluation_loop(n, model, df, returns, windowSize, prices):
    # we do somes steps of 2000
    m = 2000
    o = 500
    N = windowSize

    p_t = np.array([])

    # we compile the function with numba
    compiled_loop = jit('f4[:](f4[:],f4[:],f4[:])', nogil=True)(
        loop_without_stop_loss
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

    prices = pd.read_csv('./data/EURCHF.csv', delimiter=';', header=None)
    prices = pd.DataFrame(prices[1].values)
    prices.columns = ["ask"]
    n = prices.shape[0]

    p_t = algorithme(prices, windowSize, n, model)
