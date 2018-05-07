import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


if __name__ == '__main__':
    sns.set()

    df = pd.read_csv("../2004-2005/EURUSD-2004_01_01-2005_01_01.csv", delimiter=",")
    df.columns = [
        "DateTime Stamp", "Bar OPEN Bid Quote", "Bar HIGH Bid Quote",
        "Bar LOW Bid Quote", "Bar CLOSE Bid Quote"
    ]

    ft = pd.read_csv("../ft.csv")
    pt = pd.read_csv("../pt.csv")
    sub_df = df.iloc[2000:2000 + ft.shape[0]]
    dates = df["DateTime Stamp"].iloc[2000:2000 + ft.shape[0]]

    sub_df['ft'] = pd.Series(ft.values.transpose()[0], index=sub_df.index)
    sub_df['pt'] = pd.Series(pt.values.transpose()[0], index=sub_df.index)

    sub_df['DateTime Stamp'] = pd.to_datetime(sub_df['DateTime Stamp'])
    sub_df['DateTime Stamp'] = sub_df['DateTime Stamp'].dt.strftime('%d/%m/%y %H:%M')

    # on génère la liste de _labels_ qui seront à ''Matplo
    tick = [''] * len(dates)
    # on définit le sous-ensemble que l'on veut prendre ici: tout les 5
    size = int(len(dates) / 15)
    tick = [''] * len(sub_df['DateTime Stamp'])
    # on définit le sous-ensemble que l'on veut prendre ici: tout les 5
    tick = [item for item in sub_df['DateTime Stamp'][::size]]

    fig, axes = plt.subplots(nrows=3, ncols=1)

    axes[0].set_title("EUR/USD")
    sub_df.plot(
        'DateTime Stamp', 'Bar OPEN Bid Quote', ax=axes[0],
        legend=False, fontsize=6
    )
    plt.sca(axes[0])
    plt.xticks(np.arange(0, len(dates), size), tick, rotation=11.75)
    plt.xlabel("")

    axes[1].set_title("Signal")
    sub_df.plot('DateTime Stamp', 'ft', ax=axes[1], legend=False, fontsize=6)
    plt.sca(axes[1])
    plt.xticks(np.arange(0, len(dates), size), tick, rotation=11.75)
    plt.xlabel("")

    axes[2].set_title("P&L")
    sub_df.plot('DateTime Stamp', 'pt', ax=axes[2], legend=False, fontsize=6)
    plt.sca(axes[2])
    plt.xticks(np.arange(0, len(dates), size), tick, rotation=11.75)
    plt.xlabel("")

    plt.tight_layout(pad=0.05, h_pad=0.3, w_pad=0.3)
    plt.show()

    # max_range = 300000
    # step = 60000
    # for ele in np.arange(0, max_range, step):

    #     tmp_sub = sub_df.iloc[ele:min(ele + step, max_range)]
    #     size = int(step / 15)
    #     tick = [''] * len(tmp_sub['DateTime Stamp'])
    #     # on définit le sous-ensemble que l'on veut prendre ici: tout les 5
    #     tick = [item for item in tmp_sub['DateTime Stamp'][::size]]

    #     fig, axes = plt.subplots(nrows=3, ncols=1)

    #     axes[0].set_title("EUR/USD")
    #     sub_df.iloc[ele:min(ele + step, max_range)].plot(
    #         'DateTime Stamp', 'Bar OPEN Bid Quote', ax=axes[0],
    #         legend=False, fontsize=6
    #     )
    #     plt.sca(axes[0])
    #     plt.xticks(np.arange(0, step, size), tick, rotation=11.75)
    #     plt.xlabel("")

    #     axes[1].set_title("Signal")
    #     sub_df.iloc[ele:min(ele + step, max_range)].plot(
    #         'DateTime Stamp', 'ft', ax=axes[1],
    #         legend=False, fontsize=6
    #     )
    #     plt.sca(axes[1])
    #     plt.xticks(np.arange(0, step, size), tick, rotation=11.75)
    #     plt.xlabel("")

    #     axes[2].set_title("P&L")
    #     sub_df.iloc[ele:min(ele + step, max_range)].plot(
    #         'DateTime Stamp', 'pt', ax=axes[2],
    #         legend=False, fontsize=6
    #     )
    #     plt.sca(axes[2])
    #     plt.xticks(np.arange(0, step, size), tick, rotation=11.75)
    #     plt.xlabel("")

    #     plt.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3)
    #     plt.show()
