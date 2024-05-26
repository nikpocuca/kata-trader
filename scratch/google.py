import yfinance as yf
import matplotlib.pyplot as plt 
import ruptures as rpt
import numpy as np
import imageio 
import os 


import time 
# Set the start and end date
start_date = '2020-01-01'
end_date = '2022-01-01'

# Set the ticker
ticker = 'GOOGL'

# Get the data
data_full = yf.download(ticker, start_date, end_date)


for i in np.arange(20,data_full.shape[0]):

    data = data_full.iloc[:i]

    signal = np.diff(data["Open"].to_numpy())
    algo = rpt.Pelt(model="rbf").fit(signal)
    result = np.array(algo.predict(pen=1.5)) + 1 # magic one because of np diff

    def calculate_trend_line(x: np.array, changepoints: np.array): 
        """
            calculates trend in periods. 
        """

        # segment population x according to changepoints. 
        
        splitted_data = np.split(x, changepoints)
        
        trend_line_functions = [] 

        # loop through all the splits. 
        for i in range(len(splitted_data)): 
            
            if i == 0:
                # set up split
                t = np.arange(splitted_data[i].shape[0])
            else: 
                t = np.arange(splitted_data[i].shape[0]) + changepoints[i-1]

            price = splitted_data[i]
            
            # calculate function. 
            coef = np.polyfit(t,price,1)
            poly1d_fn = np.poly1d(coef) 

            # collect functions.
            trend_line_functions += [poly1d_fn] 


        return trend_line_functions

    trend_line_functions = calculate_trend_line(data["Open"].to_numpy(), result[:-1])

    t = np.arange(len(data["Open"].to_numpy()))


    def plot_trend_lines(t: np.array, trend_lines: list, changepoints: np.array):
        """
            take in t, changepoints, and trend lines, and plot them!
        """

        # split ts. 
        ts_split = np.split(t, changepoints)

        for i in range(len(ts_split)): 
            t_i = ts_split[i]
            plt.plot(t_i, trend_lines[i](t_i), '--k')
        

    # plot trend lines!
    changepoints = result[:-1]
    rpt.display(data["Open"].to_numpy(), list(result))
    plot_trend_lines(t, trend_line_functions, changepoints)

    plt.savefig(f"images/figure_{i}.png")


print('Creating gif\n')
gif_name = "google_algo"
file_names = os.listdir("images")

with imageio.get_writer(f'{gif_name}.gif', mode='I') as writer:
    for i in range(20, data_full.shape[0]):
        image = imageio.imread(f"images/figure_{i}.png")
        writer.append_data(image)

print('Gif saved\n')