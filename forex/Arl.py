import numpy as np
# to set an defined behavior
np.random.seed(0)
import tensorflow as tf
tf.set_random_seed(0)

import tqdm
from numba import jit
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN
from keras import backend as K
import pandas as pd


class Arl():
    """The automatic reinforcement learning class"""

    def __init__(self, windowSize=20):
        """This init our class

        The window size, compile with numba our loop and init
        the keras net

        :param windowSize: the number of elements we care.
        :type windowSize: int
        """
        self.windowSize = windowSize
        self.model = self.__init_model__(windowSize)
        self.compiled_loop = jit('f4[:](f4[:],f4[:],f4)', nogil=True)(
            self.__loop__
        )

    def __init_model__(self, N):
        """Init the Keras neural net.

        :param N: the number of features for the input vector
        :type N: int
        """
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

    def run(self, prices, n):
        """Run the learning and test phases.

        :param prices: the prices we load
        :type prices: pd.DataFrame
        :param n: the number of prices we want to back-test
        :type n: int
        """
        # compute the returns
        returns = prices.pct_change().fillna(0)
        # update the indexs
        returns = returns.set_index(returns.index - 1)

        # the df.iloc[0] is the time 0 to time windowSize - 1
        # the classe of the line is returns[0 + windowSize]
        df = self.__rolling_window__(returns['ask'].values, self.windowSize)

        # run the train-test loop
        p_t = self.__evaluation_loop__(n, df, returns['ask'].values)
        # return the p&l
        return p_t

    def __evaluation_loop__(self, n, df, returns):
        """The train-test loop

        We cut our returns vector into train-test sets. We train on train sets,
        test on the tests sets. We keep in memory the cumulated profit

        :param n: the max index of returns
        :param n: int
        :param df: the window of returns
        :param df: pd.DataFrame
        :param returns: the class for each window of df
        :param returns: np.array
        """
        # we do somes steps of 2000
        m = 2000
        o = 500
        N = self.windowSize

        p_t = np.array([1])

        for i in tqdm.tqdm(np.arange(m, n, o)):

            # first part : we treat the data
            # we select the train and test set
            train = np.array(df[i - m:i])
            test = np.array(df[i:i + o])

            # train = np.reshape(train, train.shape + (1,))
            # test = np.reshape(test, test.shape + (1,))
            # we train our model
            self.model.train_on_batch(train, returns[i - m:i])
            # we compute the classes
            #################################################################
            # we can not transform the res and use it with a threshold TODO #
            #################################################################
            y_res = np.sign(
                np.array(self.model.predict_on_batch(test)).transpose()[0]
            )
            # now we compute the cumulative profit
            p_t = np.append(
                p_t,
                self.compiled_loop(
                    y_res, returns[i + N: i + o + N - 1], p_t[-1])
            )

        # return np.cumsum(p_t)
        return p_t

    # magic
    def __rolling_window__(self, a, window):
        """It's some magic

        .. todo:: understand this part
        """
        shape = (a.shape[0] - window + 1, window) + a.shape[1:]
        strides = (a.strides[0],) + a.strides
        windows = np.lib.stride_tricks.as_strided(
            a, shape=shape, strides=strides
        )
        return pd.DataFrame(np.squeeze(windows))

    def __loop__(self, y_res, test_classes, pnl):
        """Compute the p&l using the computed class.

        :param y_res: the computed class
        :type y_res: np.array
        :param test_classes: the result class
        :type test_classes: np.array
        :param pnl: the previous p&l
        :type pnl: float
        """
        res = np.zeros(y_res.shape[0] - 2)
        i = 0
        rate = 0.01

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
