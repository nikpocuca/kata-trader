"""
stream preliminary analysis 

just import a stream .json file and see if a message can be parsed. 

"""

import argparse
import pprint

from src.kata_alpaca_engine import IngestionEngine

def main():
    """
    uses an argparse to check out a stream file
    """
    parser = argparse.ArgumentParser('stream-parser') 
    parser.add_argument('--streamfile',type=str)
    args = parser.parse_args() 

    data_list = IngestionEngine.parse_stream_data(args.streamfile)

    # data results are now in an ordered list
    pprint.pprint(data_list[0:5])

if __name__ == '__main__':
    main()