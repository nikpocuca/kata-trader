"""
kata_alpaca_utilities.py 

utility functions and classes that integrate 
with the kata engine. 

"""

KATA_DATE_FORMAT = "%Y-%m-%d"
KATA_DATE_TIME_FORMAT = "%Y-%m-%d:<%H:%M:%S>"
PATH_SECRETS = ".secrets"

import os
from datetime import datetime
import json
import getpass
import shutil
import yaml

from .logging_utilities import create_logger

logger = create_logger("engine-utilities")

from typing import List, Tuple, Callable, Union

import numpy as np
from datetime import datetime
from ..kata_models.GaussianProcessModel import RegressionModel

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import QuantLib as ql 



def create_andor_delete_dir(dir: os.PathLike = "./streams", delete: bool = True):
    """
    creates directory and deletes old one based on the name.
    """

    if os.path.exists(dir):
        if delete:
            logger.info(f"{dir} exists, deleting and removing tree")
            shutil.rmtree(dir)
            os.mkdir(dir)
        else:
            pass
    else:
        logger.info(f"{dir} does not exist, creating directory")
        os.mkdir(dir)


def create_stream_file(date: datetime, stream_dir: os.PathLike) -> os.PathLike:
    """
    Stream file creation function for logging results and stream between
    alpaca.
    """

    stream_file_name = f"stream-{date.strftime(KATA_DATE_FORMAT)}"
    path_dir = os.path.join(stream_dir, stream_file_name)

    # if file exists, than use verbose naming convention.
    if os.path.exists(path_dir + ".json"):
        logger.info("Archival mode activated, adding more stream logs")
        stream_file_name = f"stream-{date.strftime(KATA_DATE_TIME_FORMAT)}"
        path_dir = os.path.join(stream_dir, stream_file_name)

    # create file
    with open(path_dir + ".json", "w") as f:
        json.dump(
            {
                "message": f"Genesis: creation of stream file on {date.strftime(KATA_DATE_TIME_FORMAT)}",
                "user": getpass.getuser(),
                "hostname": os.uname()[1],
            },
            f,
            indent=4,
        )

    return path_dir


def acquire_credentials(secret_path: str = None):
    """
    looks for the credentials in a secret file
    """
    if secret_path is None:
        secret_path = PATH_SECRETS
    else:
        pass

    if os.path.exists(secret_path):

        # open file to secrets
        with open(secret_path, "r") as f:

            # use yaml to safely load
            key_information = yaml.safe_load(f)
            return key_information["api_key"], key_information["api_secret"]

    else:
        logger.critical("Secrets file not found")
        raise FileNotFoundError(f"File not found for {secret_path}")


def download_data(secrets_path, start_date, symbols="NVDA"):
    """
    downloads data from the alpaca
    """
    api_key, api_secret = acquire_credentials(secrets_path)

    client = StockHistoricalDataClient(api_key=api_key, secret_key=api_secret)

    request_params = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=datetime.strptime(start_date, "%Y-%m-%d"),
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df
    timestamps = df.index.map(lambda x: x[1]).values
    closing_prices = df["close"].values

    interim_df = pd.DataFrame({"timestamp": timestamps, "price": closing_prices})

    return interim_df


class BackTestData:
    """
    class to handle back testing of data.

    """

    data_full: Tuple[np.ndarray, np.ndarray]
    data: List[Tuple[np.ndarray, np.ndarray]]
    prediction_data: List[np.ndarray]
    prediction_data_price: List[np.ndarray]
    model_fits: List[RegressionModel]
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    df: pd.DataFrame

    def __init__(self, df: pd.DataFrame):
        """ """

        self.data_full = df["timestamp"].to_numpy(), df["price"].to_numpy()
        self.model_fits = []
        self.data = []
        self.df = df
        self.prediction_data_price = []
        self.prediction_data = []

        self.prediction_fit = []
        self.prediction_fit_sigma = []

    @property
    def dates_full(self):
        return pd.to_datetime(self.data_full[0]).to_pydatetime()

    @property
    def prices_full(self):
        return self.data_full[1]

    @staticmethod
    def convert_ql_to_datetime(date: datetime) -> ql.Date:
        return datetime(date.year(), date.month(), date.dayOfMonth())

    @staticmethod
    def convert_datetime_to_ql(date: datetime) -> ql.Date:
        return ql.Date(date.day, date.month, date.year)

    def create_projections(self, start_date: ql.Date, end_date: ql.Date) -> List:
        # Define today's date and the end date

        # Generate business days
        business_days = []
        current_date = start_date

        while current_date <= end_date:
            if self.calendar.isBusinessDay(current_date):
                business_days.append(current_date)
            current_date = self.calendar.advance(current_date, ql.Period(1, ql.Days))

        # Convert QuantLib Dates to datetime objects
        business_days_datetime = [
            datetime(
                current_date.year(), current_date.month(), current_date.dayOfMonth()
            )
            for current_date in business_days
        ]

        projected_datetimes = []
        # Print the datetime objects
        for dt in business_days_datetime:
            projected_datetimes.append(dt)

        return projected_datetimes

    def create_backtest_data(
        self, _from: datetime, _to: datetime, predict_to: ql.Date, store_prices=False
    ):
        """ """
        # get subset of data

        dates = self.dates_full
        prices = self.prices_full

        date_projected = self.create_projections(
            start_date=self.convert_datetime_to_ql(_to), end_date=predict_to
        )

        date_projected = np.array(date_projected)

        subset_key = (_from <= dates) & (dates <= _to)

        self.data.append((np.array(dates[subset_key]), prices[subset_key]))

        self.prediction_data.append(date_projected)

        if store_prices:
            date_set = set([dt.date() for dt in date_projected])
            matching_prices = self.df[self.df["timestamp"].dt.date.isin(date_set)][
                "price"
            ]
            self.prediction_data_price.append(matching_prices.values)

    def fit_model(self, dataset_index: Union[int, str], model_function: Callable):

        dates, prices = self.data[dataset_index]
        dates_array = np.array(
            list(map(lambda x: float(x.days), np.array(dates - dates[0])))
        )

        model_base = model_function(dates_array, prices)
        model_base.fit()

        self.model_fits.append(model_base)

    def predict(self, dataset_index: Union[int, str]):

        dates = self.prediction_data[dataset_index]
        anchor = self.data[dataset_index][0][0]
        dates = np.array(
            list(map(lambda x: float(x.days), np.array(np.array(dates) - anchor)))
        )

        dates, yfit, ysigma = self.model_fits[dataset_index].predict(Xt=dates)
        return yfit, ysigma

    def training_bands(self, dataset_index: Union[int, str], band_factor=1.96):

        dates, _ = self.data[dataset_index]
        dates_array = np.array(
            list(map(lambda x: float(x.days), np.array(dates - dates[0])))
        )
        dates, yfit, ysigma = self.model_fits[dataset_index].predict(Xt=dates_array)
        yfit, lower, upper = self.create_bands(yfit, ysigma, band_factor=band_factor)
        return yfit, lower, upper

    def create_bands(self, yfit, ysigma, band_factor=1.96):
        lower_bound = yfit - ysigma * band_factor
        upper_bound = yfit + ysigma * band_factor
        return yfit, lower_bound, upper_bound
