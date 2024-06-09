"""
change_point.py version 0.1.0 

Contains class on changepoint and methods related to attaining changepoint information. 

"""

import ruptures as rpt
import numpy as np 
import datetime 
from typing import List 
import os
import asyncio
import threading
from asyncio.unix_events import _UnixSelectorEventLoop as EventLoop
from threading import Thread

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
    data_list_raw: List[float]  = []
    penalty: float
    logger: Logger

    name: str 
    sub_dir_name: str
    image_name_list: List[str] = [] 
    image_counter: int = 0

    rpt_model: Pelt
    fit_timestamp: TimeStamp
    has_fit: bool = False
    rpt_results: np.ndarray
    changepoint_thread: Thread

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

    def __init__(self, 
                 output_dir_path: str,
                   name: str = '',
                   model: str = 'rbf',
                   penalty: float = 1.0) -> None:
        """
        creates directory if it doesnt exist, 
        then creates a sub-directory for the output path
        """

        if os.path.exists(output_dir_path):
            
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

        # create logger 
        cp_logger = create_logger(f'{self.name}-changepoint-logger')
        self.logger = cp_logger
        self.logger.info("changepoint model start")

        # define model. 
        self.rpt_model = rpt.Pelt(model=model)
        self.penalty = penalty

        # launch changepoint daemon
        self.run_model()

    async def run_model(self): 
        """
        begins the async call to start looking through the data for changepoints
        """

        while True:
            
            if self.data_len == 0:
                continue
            else: 

                # FIT MODEL PROCEDURE
                
                # check to see if the model has been fitted. 
                if not self.has_fit:
                    self.fit_rpt_model()
                    
                # if the model has, go through checks                
                else: 

                    # check to see if the number of samples have changed. 
                    if len(self.deltas) != self.fit_samples:
                        
                        # refit if they have not changed.
                        self.fit_rpt_model()

                    else:
                        # if number of samples are the same dont do anything
                        continue 
                        
    def fit_rpt_model(self):
        """
        fits model using rutpures on the deltas data. 
        """
        self.rpt_model.fit(self.deltas)
        self.fit_timestamp = datetime.datetime.now().strftime(KATA_DATE_TIME_FORMAT)
        self.fit_results = np.array(self.rpt_model.predict(pen=self.penalty)) + MAGIC_OFFSET


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
