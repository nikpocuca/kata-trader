import ruptures as rpt 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from src.kata_cp.plot_utils import display

def main(): 
    df = pd.read_csv('ibm_prices.csv')

    prices = df['price'].values
    signal = np.diff(prices)
    algo = rpt.Pelt(model='rbf').fit(signal)

    predicted = algo.predict(pen=0.5)
    result = np.array(predicted) + 1
    changepoints = list(result)

    fig, ax = display(prices, changepoints, figsize=(10, 6), alpha=0.80)

    plt.savefig('ibm.png')



if __name__ == '__main__':
    main()


