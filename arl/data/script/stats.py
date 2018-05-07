ptu = pd.read_csv('pt_usdjpy_2006-2010.csv')
pte = pd.read_csv('pt_eurusd_2006-2010.csv')
ptg = pd.read_csv('pt_gbpusd_2006-2010.csv')

precision = 2
width = 4

for currency, pt in zip(['usdjpy', 'eurusd', 'gbpusd'], [ptu, pte, ptg]):
    print(f"{currency}")
    for annees, array in zip(['2006', '2007', '2008', '2009', '2010'], np.array_split(pt.values.transpose()[0],5)):
        i = np.argmax(np.maximum.accumulate(array) - array)
        j = np.argmax(array[:i])
        print(f"\hline\n\t{annees} & {round(array[j], 5)} & {round(array[i], 5)} & {round(array[j] - array[i], 5)} & {round(array[-1] - array[0], 5)} & {round(array.mean(), 5)} & {round(array.var(), 5)}\\\\")
