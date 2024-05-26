"""
kata_alpaca_engine

A class containing all that is required to connect,
integrate and facilitate all connections to 
alpaca trading platform. 

Facilitates data transfer and request. 

"""

import os
from datetime import datetime
import json

from .logging_utilities import create_logger

logger = create_logger("engine-internal")

# alpaca related
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream
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

STREAM_DIRECTORY = "streams"


class KataAlpacaEngine:
    """
    KataAlpacaEngine

    creates connections, streams to the alpaca api system.

    """

    api_key: str
    api_secret: str
    stream_file_path: os.PathLike

    paca_client: TradingClient
    paca_stream: TradingStream
    paca_stream_loop: EventLoop
    paca_stream_thread: Thread
    paca_stream_process: Process

    def __init__(self, archive_mode: bool = True, secret_path: str = None) -> None:
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

        # begin stream into directory
        stream = TradingStream(
            api_key=self.api_key, secret_key=self.api_secret, paper=True
        )
        self.paca_stream = stream
        self.start_stream_in_background()

    def start_stream_in_background(self):
        """
        starts a stream in the background
        """

        # set the loop
        self.paca_stream_loop = asyncio.get_event_loop()
        logger.info("Stream connection established, watching for data.")
        self.paca_stream.subscribe_trade_updates(self.trade_stats)
        self.paca_stream_thread = threading.Thread(
            target=self.background_stream_job, args=(self.paca_stream_loop,)
        )
        self.paca_stream_thread.daemon = True
        self.paca_stream_thread.start()

    def background_stream_job(self, loop: EventLoop):
        """
        run the actual job in the background.
        """
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.paca_stream.run())

    def __del__(self):
        """
        close the stream in case something
        """
        if self.paca_stream._ws:
            self.paca_stream.close()

    async def trade_stats(self, data):
        with open(self.stream_file_path + ".json", "a") as f:
            data_dump = json.loads(data.json())
            order_info = data_dump["order"]
            logger.info(
                f"{order_info['symbol']} | {order_info['side']}-{order_info['qty']} logging trade data into file"
            )
            json.dump(data_dump, f, indent=4)
