"""
change_point.py version 0.1.0 

Contains class on changepoint and methods related to attaining changepoint information. 

"""

import ruptures as rpt
import numpy as np 
import datetime 
from typing import List, Callable
import os
import asyncio
import threading
from asyncio.unix_events import _UnixSelectorEventLoop as EventLoop
from threading import Thread
import matplotlib.pyplot as plt
import shutil

from ..kata_alpaca_engine.ingestion_engine import IngestionEngine

from ..kata_alpaca_engine.engine_utilities import (
    KATA_DATE_TIME_FORMAT,
    KATA_DATE_FORMAT,
)

from ..kata_alpaca_engine.logging_utilities import (
    create_logger,
    Logger
)

from ruptures.detection.pelt import Pelt

"""
TimeStamp type definition is basically KATA_DATE_TIME_FROMAT 
in str form 
"""
TimeStamp = str
MAGIC_OFFSET=1

class KataChangePointEngine: 
    """
    encapsulates changepoint detection and calculation. 

    wraps around the ruptures package. Calculates things as a daemon
    """

    data_list_json: List[dict] = []

    _data_list_raw: List[float] 
    _data_list_raw_length: int 

    penalty: float
    logger: Logger

    name: str 
    output_dir: str
    sub_dir_name: str
    image_name_list: List[str] = [] 
    image_counter: int = 0

    rpt_model: Pelt
    fit_timestamp: TimeStamp
    has_fit: bool = False
    rpt_results: np.ndarray
    changepoint_thread: Thread

    trend_lines: List

    fitted_data: List[float]
    fitted_deltas: List[float]
    parsing_function: Callable

    @property
    def data_list_raw(self):
        """
        parses json list raw. 
        """

        if not hasattr(self, '_datalist_raw'):
            self._data_list_raw = [self.parsing_function(entry) for entry in self.data_list_json if not IngestionEngine.is_genesis(entry)]
            self._data_list_raw_length = len(self._data_list_raw)
            return self._data_list_raw

        # minus one because of genesis 
        elif self._data_list_raw_length != (len(self.data_list_json) -1):
            self._data_list_raw = [self.parsing_function(entry) for entry in self.data_list_json]
            self._data_list_raw_length = len(self._data_list_raw)
            return self._data_list_raw

        else:
            return self._data_list_raw

    
    @property
    def data(self): 
        """
        converts clean float data into a tensor. 
        """
        return np.array(self.data_list_raw)

    @property
    def data_len(self):
        """
        easy return for data length as this tends to change a lot. 
        """
        return len(self.data_list_raw)

    @property
    def fit_samples(self):
        """
        easy return for number of samples that have been fit.

        warning, may return as a result  
        """
        return self.rpt_model.n_samples

    @property
    def deltas(self):
        """
        calculate differences in numpy form
        """
        return np.diff(self.data_list_raw)
    
    @staticmethod
    def deltas_calc(x): 
        """
        calculates deltas instead of as property, useful only for fitted data 
        """
        return np.diff(x)

    def __init__(self, 
                 output_dir_path: str,
                 parsing_function: Callable,
                   name: str = '',
                   model: str = 'rbf',
                   penalty: float = 1.0) -> None:
        """
        creates directory if it doesnt exist, 
        then creates a sub-directory for the output path
        """

        if os.path.exists(output_dir_path):
            self.output_dir = output_dir_path
            # create sub-directory with special name, and time-stamp it
            if name == '':
                sub_dir_name = f'images-{datetime.datetime.now().strftime(KATA_DATE_TIME_FORMAT)}' 
                self.name = 'unknown'
            else: 
                sub_dir_name = f'{name}-images-{datetime.datetime.now().strftime(KATA_DATE_TIME_FORMAT)}' 
                self.name = name

            self.sub_dir_name = sub_dir_name
        
        else: 
            os.mkdir(output_dir_path)
            self.output_dir = output_dir_path

        full_path = os.path.join(self.output_dir, self.sub_dir_name)
        if not os.path.exists(full_path):
            os.mkdir(full_path)

        self.parsing_function = parsing_function

        # create logger 
        cp_logger = create_logger(f'{self.name}-changepoint-logger')
        self.logger = cp_logger
        self.logger.info("changepoint model start")

        # define model. 
        self.rpt_model = rpt.Pelt(model=model)
        self.penalty = penalty

        # # launch changepoint daemon
        # breakpoint()
        # self.run_model()

    # async def run_model(self): 
    def run_model(self): 

        """
        begins the async call to start looking through the data for changepoints
        """
        while True:
            if self.data_len <= 5:
                continue
            else: 

                # FIT MODEL PROCEDURE
                
                # check to see if the model has been fitted. 
                if not self.has_fit:
                    self.fit_rpt_model()
                    self.has_fit = True
                    self.plot_trend_lines()
                    
                # if the model has, go through checks                
                else: 
                    
                    # check to see if the number of samples have changed. 
                    if len(self.deltas) != self.fit_samples:

                        # refit if they have not changed.
                        self.fit_rpt_model()
                        self.plot_trend_lines()

                    else:
                        # if number of samples are the same dont do anything
                        continue 
                        
    def fit_rpt_model(self):
        """
        fits model using rutpures on the deltas data. 
        """
        self.logger.info('fitting')
        self.fitted_data = np.array(self.data_list_raw)
        self.fitted_deltas = self.deltas_calc(self.fitted_data)
        self.rpt_model.fit(self.fitted_deltas)
        self.fit_timestamp = datetime.datetime.now().strftime(KATA_DATE_TIME_FORMAT)
        self.fit_results = np.array(self.rpt_model.predict(pen=self.penalty)) + MAGIC_OFFSET
        self.calculate_trend_line()


    def calculate_trend_line(self):
        """
        calculates the trend line based on the the splitting of multiple changepoints
        TODO(attach timestamp deltas possible)
        """

        # segment population x according to changepoints.
        x = self.fitted_data
        changepoints = self.fit_results[:-1] 
        
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

        self.trend_lines = trend_line_functions
    

    def plot_trend_lines(self):
        """
        plots trend line to file based on image count and filepath
        """
        plt.clf() 

        t = np.arange(len(self.fitted_data))
        trend_lines = self.trend_lines
        changepoints = self.fit_results[:-1]

        ts_split = np.split(t, changepoints)

        for i in range(len(ts_split)): 
            t_i = ts_split[i]
            plt.plot(t_i, trend_lines[i](t_i), '--k')

        changepoints = self.fit_results[:-1]
        rpt.display(self.fitted_data, self.fit_results)

        # image name output 
        image_name_output = os.path.join(self.output_dir, 
                                         self.sub_dir_name, 
                                         f'{self.name}_{self.image_counter}.png')
        
        current_image_output = os.path.join(self.output_dir, 
                                         self.sub_dir_name, 
                                         'current_image.png')
        plt.savefig(image_name_output)
        plt.close()
        shutil.copy(image_name_output, current_image_output)

        self.image_counter += 1
        self.image_name_list.append(image_name_output) 
        plt.clf()


    def start_changepoint_model_in_background(self) -> None:
        """
        starts a changepoint model in the background 
        """

        # set the loop
        self.stock_stream_loop = asyncio.get_event_loop()
        self.logger.info("changepoint starting.")

        self.changepoint_thread = threading.Thread(
            target=self.background_job, args=(self.stock_stream_loop,)
        )

        self.changepoint_thread.daemon = True # if main thread dies, then this dies as well. 
        self.changepoint_thread.start()

    def background_job(self, loop: EventLoop) -> None:
        """
        run the actual job in the background.
        """
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.run_model())
