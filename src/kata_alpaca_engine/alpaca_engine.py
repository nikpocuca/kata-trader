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

logger = create_logger('engine logger')

# alpaca related
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream

# class related
from .engine_utilities import (
    create_andor_delete_dir,
    create_stream_file,
    acquire_credentials
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

    def __init__(self, archive_mode: bool = True) -> None:
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
        self.api_key, self.api_secret = acquire_credentials()

        # attempt to create connection to client. 
        try: 
            self.paca_client = TradingClient(self.api_key, self.api_secret)
            logger.info('Alpaca trading client connected.')
        except: 
            bad_connection_message = 'Alpaca trading client refused to connect'
            logger.critical(bad_connection_message)
            raise ConnectionError(bad_connection_message)

        # begin stream into directory
        stream = TradingStream(api_key=self.api_key, secret_key=self.api_secret, paper=True)
        stream.subscribe_trade_updates(self.trade_stats)

        logger.info("Stream connection established, watching for data.")
        stream.run()

    async def trade_stats(self, data):
        with open(self.stream_file_path + '.json', 'a') as f:
            data_dump =json.loads(data.json())
            order_info = data_dump['order']
            logger.info(f"{order_info['symbol']} | {order_info['side']}-{order_info['qty']} logging trade data into file")
            json.dump(data_dump,f, indent=4)