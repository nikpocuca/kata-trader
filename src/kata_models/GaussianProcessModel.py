from typing import List

import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as K 
import numpy as np

# define optimal kernels 

OPTIMAL_KERNELS = 1.0 * K.ConstantKernel()\
    + 1.0 * K.DotProduct()\
    + 1.0 * K.WhiteKernel(noise_level=0.5)\
    + 1.0 * K.Matern(nu=0.75)


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
        return self.Xt, y_fit, sigma

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