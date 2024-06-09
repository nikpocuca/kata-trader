# Deployment Scripts

## Stream Trader Deployment

This script deploys the stream trader in the right hand side, and tracks what occurs on the account.

``` bash
# declare path to secret
SECRET_PATH="<insert-path-to-secret>"

# export src directory to python path for now, later it will be a full package
export PYTHONPATH=$(pwd)/src:$PYTHONPATH

# deploy script, it will run as a daemon.
python3 ./scripts/stream_trades.py --config $SECRET_PATH

# output results can be found in ./streams with a datestamp. 

# deploy script for stock trade. 

export PYTHONPATH=$(pwd)/src:$PYTHONPATH; 
python3 ./scripts/stream_stocks.py --symbol TSLA \
        --config ./src/.secrets 


# deploy script for simulation 
export PYTHONPATH=$(pwd)/src:$PYTHONPATH; 
# python3 ./scripts/stream_simulation.py --file <path_to_streamed_file> --demo
python3 ./scripts/stream_simulation.py --file ./tsla-stream-2024-06-05.json --demo


```
