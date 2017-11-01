import pandas as pd
import numpy as np


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
