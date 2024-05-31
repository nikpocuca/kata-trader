"""
ingestion_engine.py

A class containing all that is required to connect,
integrate and facilitate all connections to 
alpaca trading platform. 

Facilitates data transfer and request. 

"""

from typing import List, Optional, Tuple
import os
from datetime import datetime
import json

from .logging_utilities import create_logger

logger = create_logger("ingestion-internal")

# alpaca related
from alpaca.trading.client import TradingClient
from alpaca.data.live.stock import StockDataStream
from alpaca.data.enums import DataFeed

import asyncio
from asyncio.unix_events import _UnixSelectorEventLoop as EventLoop
import threading

from threading import Thread
from multiprocessing import Process

# class related
from .engine_utilities import (
    create_andor_delete_dir,
    create_stream_file,
    acquire_credentials,
)

STREAM_DIRECTORY = "stock-data-streams"

from enum import Enum

class IngestionResults(Enum):
    SUCCESS = 0
    FAILURE = 1

class IngestionEngine:
    """
    Ingestion engine 

    creates connections, streams to the alpaca api system, and streams
    historical data. 

    """

    api_key: str
    api_secret: str
    stream_file_path: os.PathLike

    paca_client: TradingClient
    paca_stock_stream: StockDataStream
    stock_stream_loop: EventLoop
    stock_stream_thread: Thread
    stock_stream_process: Process

    formatting_dict: dict = {
        'T':'message_type',
        'S':'symbol',
        'bx': 'bid_exchange_code',
        'bp': 'bid_price',
        'bs': 'bid_size',
        't': 'time_stamp',
        'ap': 'ask_price',
    }

    def __init__(self,
                 stocks: List[str], 
                 archive_mode: bool = True, 
                 secret_path: str = None, 
                 ) -> None:
        """
        upon creation, create or delete an old stream log,
        create a new stream log to write in trading messages.

        archive_mode: boolean always set to True by default, so that
        stream messages can come in as they go.
        """

        # create stream and genesis header,
        # if we are not archiving we delete streams.
        create_andor_delete_dir(STREAM_DIRECTORY, delete=(not archive_mode))
        self.stream_file_path = create_stream_file(datetime.now(), STREAM_DIRECTORY)

        # acquire creds and create connection
        self.api_key, self.api_secret = acquire_credentials(secret_path)

        # attempt to create connection to client.
        try:
            self.paca_client = TradingClient(self.api_key, self.api_secret)
            logger.info("Alpaca trading client connected.")
        except:
            bad_connection_message = "Alpaca trading client refused to connect"
            logger.critical(bad_connection_message)
            raise ConnectionError(bad_connection_message)


        stock_stream = StockDataStream(api_key=self.api_key,
                                       raw_data=True,
                                       secret_key=self.api_secret,
                                       feed=DataFeed.IEX)
        

        self.stock_stream = stock_stream
        self.start_stream_in_background(stocks=stocks)

    @property
    def formatting_key_list(self):
        """
        formatting 
        """
        return list(self.formatting_dict.keys())

    def start_stream_in_background(self, stocks) -> None:
        """
        starts a stream in the background
        """

        # set the loop
        self.stock_stream_loop = asyncio.get_event_loop()
        logger.info("Stream connection established, watching for data.")
        self.stock_stream.subscribe_quotes(self.stock_handler, stocks)
        self.stock_stream_thread = threading.Thread(
            target=self.background_stream_job, args=(self.stock_stream_loop,)
        )
        self.stock_stream_thread.daemon = True # if main thread dies, then this dies as well. 
        self.stock_stream_thread.start()

    def background_stream_job(self, loop: EventLoop) -> None:
        """
        run the actual job in the background.
        """
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.stock_stream.run())

    def __del__(self):
        """
        close the stream in case something
        """
        if self.stock_stream._ws:
            self.stock_stream.close()

    def map_key(self, key) -> Optional[str]:
        """
        mapping the key to the new key setting, if failure throw error
        """
        new_key = self.formatting_dict.get(key, IngestionResults.FAILURE)
        if new_key == IngestionResults.FAILURE:
            raise KeyError(f"Ingestion result failed, key processed {key}")
        else: 
            return new_key

    def generate_pair(self, key: str, stream_dict: dict) -> Optional[Tuple[str,str]]:
        """
        generates pair from new mapping 
        """ 
        value = stream_dict[key]
        new_key = self.map_key(key)
        return new_key, value
    
    def update_formatted_dict_r(self, key, stream_dict, new_dict): 
        """
        updates the new dictionary with the formatted keys and values
        """
        new_key, value = self.generate_pair(key,stream_dict)
        new_dict[new_key] = value
                

    async def stock_handler(self, data):
        with open(self.stream_file_path + ".json", "a") as f:
            data_new = data.copy() 

            data_new['t'] = datetime.fromtimestamp(data['t'].seconds)
            data_new['t'] = data_new['t'].strftime("%Y-%m-%d: %H:%M:%S")

            data_formatted = {}
            self.update_formatted_dict_r('t', data_new,data_formatted)
            self.update_formatted_dict_r('bp', data_new, data_formatted)
            self.update_formatted_dict_r('ap', data_new, data_formatted)
            self.update_formatted_dict_r('S', data_new, data_formatted)
            data_formatted['message_type'] = 'quote'
            logger.info(
                f"QUOTE {data_formatted['symbol']} | ASK({data_formatted['ask_price']}) | BID({data_formatted['bid_price']})"
            )
            json.dump(data_formatted, f, indent=4)


