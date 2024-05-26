"""
script_utilities.py 

basic utility objects and functions that assist with launching workloads in general 

"""

from kata_alpaca_engine.logging_utilities import create_logger
from enum import Enum
import logging
from argparse import ArgumentParser


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
