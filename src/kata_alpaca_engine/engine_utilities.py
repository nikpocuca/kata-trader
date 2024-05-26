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
