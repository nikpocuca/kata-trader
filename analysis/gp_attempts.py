import os 
from typing import List
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as K 
import matplotlib.pyplot as plt 

from matplotlib.pyplot import figure
from src.script_utilities import create_workload_parser
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

OPTIMAL_KERNELS = 1.0 * K.ConstantKernel()\
    + 1.0 * K.DotProduct()\
    + 1.0 * K.WhiteKernel(noise_level=0.5)\
    + 1.0 * K.Matern(nu=0.75)


from src.kata_alpaca_engine.engine_utilities import acquire_credentials

class RegressionModel:
    """
    regressor model class that wraps around sklearn. 
    """
    X: np.ndarray
    y: np.ndarray

    # define training data
    train_X: np.ndarray
    train_y: np.ndarray

    train_fit_y: np.ndarray
    train_fit_y_sigma: np.ndarray

    Xt: np.ndarray
    yt: np.ndarray = None
    test_X: np.ndarray
    test_y: np.ndarray
    test_fit_y: np.ndarray
    test_fit_y_sigma: np.ndarray
    kernels: List

    model: GaussianProcessRegressor

    band_factor: float = 1.96

    def __init__(self, X, y, kernels=OPTIMAL_KERNELS, random_state=0) -> None:
        self.X = X 
        self.y = y
        self.train_X = np.expand_dims(X, -1)
        self.train_y = y
        self.model = GaussianProcessRegressor(kernel=kernels, random_state=random_state)

    def fit(self, X=None, y=None):
        
        # if nothing is specified, just fit. 
        if X is None:
            self.model.fit(self.train_X, self.train_y)    
        else: 
            self.train_X = np.expand_dims(X, -1)
            self.train_y = y
            self.X = X 
            self.y = y
            self.model.fit(self.train_X, self.train_y)    

        y_fit, sigma = self.model.predict(self.train_X, return_std=True)
        self.train_fit_y = y_fit 
        self.train_fit_y_sigma = sigma 

    def predict(self, Xt=None, yt=None, next=5):
        if Xt is None:
            Xt = np.copy(self.X) 
            last_day = Xt[-1]
            next_days = np.arange(last_day+1, last_day + next+1)
            Xt = np.concatenate((Xt, next_days))
            Xt = np.expand_dims(Xt, -1)
            self.test_X = Xt
            self.test_y = None 
        else: 
            Xt = np.expand_dims(Xt, -1)
            self.test_X = Xt
            self.test_y = yt
            self.yt = yt 

        y_fit, sigma = self.model.predict(self.test_X, return_std=True)
        self.test_fit_y = y_fit 
        self.test_fit_y_sigma = sigma 
        self.Xt = self.test_X.flatten() 


    def plot_train_data(self, clf=True): 
        plt.scatter(self.X, self.y, s=10, color='black')
        plt.plot(self.X, self.train_fit_y, color='green')
        bounder = self.band_factor

        plt.fill_between(self.X, 
                        self.train_fit_y - self.train_fit_y_sigma * bounder, 
                        self.train_fit_y + self.train_fit_y_sigma * bounder,
                        color="tab:green",
                        alpha=0.2,
                        )
        
        if clf:
            plt.savefig('training_fit.png')
            plt.clf()


    def plot_test_data(self, clf=True): 
        plt.plot(self.Xt, self.test_fit_y, color='blue')
        bounder = self.band_factor
        lower = self.test_fit_y - self.test_fit_y_sigma * bounder
        upper = self.test_fit_y + self.test_fit_y_sigma * bounder
        plt.fill_between(self.Xt, 
                        self.test_fit_y - self.test_fit_y_sigma * bounder, 
                        self.test_fit_y + self.test_fit_y_sigma * bounder,
                        color="tab:blue",
                        alpha=0.2,
                        )
        
        plt.scatter(self.X, self.y, s=20, color='black')
        if self.yt is not None: 
            plt.scatter(self.Xt, self.yt, s=5, color='green', alpha=0.5)


        if clf:
            plt.savefig('test_fit.png')
            plt.clf()

        return lower, upper

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
    
    return interim_df        

def get_data():
    """
    
    """

    workload_parser = create_workload_parser('download-data')
    workload_parser.add_argument("--start", type=str, 
                                help='start timestamp from when you want to pull data', 
                                required=True)
    
    args = workload_parser.parse_args()

    df = download_data(args) 

    close = df['price'].values
    timestamps = df['timestamp'].values

    try: 
        datetime_array = np.array([datetime.strptime(date, '%Y-%m-%d %H:%M:%S') for date in timestamps])
    except: 
        datetime_array = np.array([datetime.strptime(str(date), '%Y-%m-%dT%H:%M:%S.%f000000') for date in timestamps])

    stock_split_for_nvidia_date = datetime.strptime('2024-06-08','%Y-%m-%d')
    split_check = datetime_array <= stock_split_for_nvidia_date
    close[split_check] = close[split_check] / 10

    # Calculate the number of days from the first date
    start_date = datetime_array[0]
    days_array = np.array([(date - start_date).days for date in datetime_array])
    days_array = days_array.astype('float64')
    close = close.astype('float64')

    df['close'] = close 
    df['days'] = days_array

    return df

def prepare_data_for_training(df,offset=15):

    days_array = df['days']
    close = df['close']

    train_data, test_data = close[:-offset], close[-offset:]
    train_ts, test_ts = days_array[:-offset], days_array[-offset:]

    return {
        'train': (train_ts, train_data), 
        'test': (test_ts, test_data)
    }




if __name__ == '__main__':
    print("Running GP Analysis")

    df = get_data()

    X = df['days'].to_numpy() 
    y = df['close'].to_numpy()
    # X = np.append(X, 199.0)
    # y = np.append(y, 120.0)

    figure(figsize=(10, 8), dpi=300)
    plt.tight_layout() 


    # days_pick = np.arange(0,y.shape[0]-10)
    # days_pick = np.arange(0,50)

    lower = 0
    upper = y.shape[0]
    lower_test = 0
    upper_test = y.shape[0]
    days_pick_test = np.arange(lower_test, upper_test)
    days_pick = np.arange(lower,upper)
    model = RegressionModel(X=X[days_pick],y=y[days_pick], random_state=2)
    model.band_factor = 2.576
    # model.band_factor = 3.291
    model.fit() 
    model.plot_train_data(clf=False)

    model.predict() 

    lower, upper = model.plot_test_data(clf=True)

    print(f"Lower: {lower[-1]}")
    print(f"Upper: {upper[-1]}")
