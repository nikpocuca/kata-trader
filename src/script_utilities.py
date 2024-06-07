"""
script_utilities.py 

basic utility objects and functions that assist with launching workloads in general 

"""

from kata_alpaca_engine.logging_utilities import create_logger
from enum import Enum
import logging
from argparse import ArgumentParser
from datetime import datetime 
import time 


class WorkloadResult(Enum):
    """
    WorkloadResult is an enum class that aims to
    be the result of running a main.py function for any script or workload
    """

    SUCCESS = 0
    FAILURE = 1


# define workload logger and import it later as desired.
workload_logger = create_logger("WorkloadLogger")
workload_logger.setLevel(logging.DEBUG)


def create_workload_parser(name: str) -> ArgumentParser:
    """
    creates an argument parser for parsing workload configurations.
    """
    parser = ArgumentParser(name)
    parser.add_argument("--config", required=True)
    return parser



def check_market_close(sleep_time=25): 
    """
    check to see if the market is closed according to the day, to be used
    in a while true loop as a daemon  
    """
    time.sleep(sleep_time)
    current_dt = datetime.now()

    market_close_day = current_dt.day
    market_close_year = current_dt.year
    market_close_month = current_dt.month 
    market_close_hour = 20
    market_close_minute = 0

    market_close_dt = datetime(year=market_close_year, 
                            month=market_close_month,
                            day=market_close_day, 
                            hour=market_close_hour, 
                            minute=market_close_minute)

    return current_dt >=  market_close_dt