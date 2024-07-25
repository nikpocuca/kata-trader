from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime 
import pandas as pd

from src.kata_alpaca_engine.engine_utilities import acquire_credentials
from src.script_utilities import create_workload_parser

import os
import matplotlib.pyplot as plt
import numpy as np

import src.kata_cp.EWMA as EWMA

workload_parser = create_workload_parser('download-data')
workload_parser.add_argument("--start", type=str, 
                             help='start timestamp from when you want to pull data', 
                             required=True)
workload_parser.add_argument("--download", type=bool, 
                             required=True, default=False)

def download_data(args):
    """
    downloads data from the alpaca 
    """
    api_key , api_secret = acquire_credentials(args.config)

    client = StockHistoricalDataClient(api_key=api_key,
                                       secret_key=api_secret)

    request_params = StockBarsRequest(
        symbol_or_symbols="NVDA",
        timeframe=TimeFrame.Day,
        start=datetime.strptime(args.start, '%Y-%m-%d')
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df 
    timestamps = df.index.map(lambda x : x[1]).values
    closing_prices = df['close'].values

    interim_df = pd.DataFrame({'timestamp':timestamps, 
                               'price': closing_prices})
    
    interim_df.to_csv('nvda_prices.csv', index=False)


from statsmodels.tsa.arima.model import ARIMA

def fit_arima_model(data: pd.DataFrame):
    """
    
    """


    close = data['price'].values
    timestamps = data['timestamp'].values
    datetime_array = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in timestamps])
    stock_split_for_nvidia_date = datetime.strptime('2024-06-08','%Y-%m-%d')
    split_check = datetime_array <= stock_split_for_nvidia_date
    close[split_check] = close[split_check] / 10

    # Calculate the number of days from the first date
    start_date = datetime_array[0]
    days_array = np.array([(date - start_date).days for date in datetime_array])
    days_array = days_array.astype('float64')
    close = close.astype('float64')


    # fit rolling arima model. 
    offset_split = 16
    train_data, test_data = close[:-offset_split], close[-offset_split:]
    train_ts, test_ts = timestamps[:-offset_split], timestamps[-offset_split:]
    history = [x for x in train_data]
    y = test_data
    # make first prediction
    predictions = list()
    model = ARIMA(history, order=(1,1,0))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)
    history.append(y[0])

    # rolling forecasts
    for i in range(1, len(y)):
        # predict
        model = ARIMA(history, order=(1,1,0))
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        # invert transformed prediction
        predictions.append(yhat)
        # observation
        obs = y[i]
        history.append(obs)


    # Plot results. 

    # Convert timestamp strings to datetime objects
    timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
    timestamps = np.array(timestamps)

    test_ts = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in test_ts]
    test_ts = np.array(test_ts)


    # plot model vs truth. 
    
    # training data
    plt.plot(timestamps[-offset_split - 5:-offset_split], history[-offset_split - 5:-offset_split], c="black", label="training")
    plt.scatter(test_ts, predictions, c='purple', label='model fit')
    plt.plot(test_ts, test_data, c='red', label='real')

    # # Customize the x-axis ticks
    # num_ticks = 5  # Number of ticks you want to show
    # step = max(1, len(timestamps) // num_ticks)  # Calculate the step size
    # plt.xticks(ticks=timestamps[::step], labels=[ts.strftime('%Y-%m-%d') for ts in timestamps[::step]], rotation=45)

    plt.savefig('arima_model_fit.png')
    

def fit_changepoint_model(data: pd.DataFrame): 
    """
    
    """

    # define X,Y

    close = data['price'].values
    timestamps = data['timestamp'].values

    # Convert the array of timestamps to datetime objects

    datetime_array = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in timestamps])

    stock_split_for_nvidia_date = datetime.strptime('2024-06-08','%Y-%m-%d')
    
    split_check = datetime_array <= stock_split_for_nvidia_date

    close[split_check] = close[split_check] / 10


    # Calculate the number of days from the first date
    start_date = datetime_array[0]
    days_array = np.array([(date - start_date).days for date in datetime_array])


    days_array = np.expand_dims(days_array,-1)
    close = np.expand_dims(close, -1)
    days_array = days_array.astype('float64')
    close = close.astype('float64')

    sd = np.std(close)

    # close = close - np.mean(close)
    # data = list((((close - np.mean(close))/ sd)).flatten())
    data = list(close.flatten())
    model = EWMA(r=0.75, L=1.5, burnin=1, mu=data[0], sigma=1.0)


    model.process(data=data)

    plt.figure(figsize=(15, 8))


    # Convert timestamp strings to datetime objects
    timestamps = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') for ts in timestamps]
    timestamps = np.array(timestamps)
    # Create the scatter plot
    plt.scatter(timestamps, data, c="b", s=20, alpha=0.2)

    # Customize the x-axis ticks
    num_ticks = 5  # Number of ticks you want to show
    step = max(1, len(timestamps) // num_ticks)  # Calculate the step size
    plt.xticks(ticks=timestamps[::step], labels=[ts.strftime('%Y-%m-%d') for ts in timestamps[::step]], rotation=45)

    
    plt.plot(timestamps, model.Z, c="m", label="$Z$")
    plt.plot(timestamps, model._mu, c="red", label="$Z$")
    plt.plot(timestamps, np.asarray(model._mu) + np.asarray(model.sigma_Z) * model.L,
            c="y", label="$\mu \pm L \sigma_Z$")
    plt.plot(timestamps, np.asarray(model._mu) - np.asarray(model.sigma_Z) * model.L,
            c="y")
    plt.scatter(timestamps[np.array(model.changepoints)], np.asarray(model.Z)[model.changepoints], marker="v",
                label="Alarm", color="green", zorder=10)
    plt.xlabel("Time")
    plt.ylabel("Observations")
    plt.legend()
    plt.tight_layout()
    plt.savefig('changepoint_results.png')
    plt.clf()



def main():
    """
    download data workload ofr some stock price example     
    """
    args = workload_parser.parse_args()

    # download 
    if args.download: 
        print("hit")
        download_data(args)

    if os.path.exists(os.path.join('nvda_prices.csv')): 
        df = pd.read_csv('nvda_prices.csv')
        model = fit_changepoint_model(df)
        model = fit_arima_model(df)


if __name__ == '__main__':
    main()