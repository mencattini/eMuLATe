import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def read_csv(n, filename='/home/romain/gitFile/eMuLATe/arl/data/EURUSD.dat'):
    with open(filename) as f:
        elements = []
        # we split each line in ask and bid
        for ele in f.readlines():
            ask_bid = ele.split(' ')[2]
            splited = ask_bid.split('/')
            ask, bid = splited[0], splited[1]
            elements.append(np.array([np.double(ask), np.double(bid)]))
    res = pd.DataFrame(elements, columns=["ask", "bid"])
    res.to_hdf(f'EURUSD_{n}.hdf', 'eurusd')
    return res


def read_hdf(n):
    filename = f'./data/EURUSD_{n}.hdf'
    return pd.read_hdf(filename, key='eurusd')


def plot_all(
        p_t, prices, n, title_one="$Cumulated_{profit}$", title_two="EUR/USD"):
    sns.set_style("darkgrid")

    plt.subplot(2, 1, 1)
    plt.plot(p_t)
    plt.grid(True)
    plt.title(title_one)
    plt.xlabel("ticks")
    plt.ylabel("profit")
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    plt.plot(prices[0:n])
    plt.grid(True)
    plt.title(title_two)
    plt.xlabel("ticks")
    plt.ylabel("prices")
    plt.tight_layout()
    plt.show()
