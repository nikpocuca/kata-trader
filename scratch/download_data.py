from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime 
import pandas as pd

from src.kata_alpaca_engine.engine_utilities import acquire_credentials
from src.script_utilities import create_workload_parser


workload_parser = create_workload_parser('download-data')

def main():
    """
    download data workload ofr some stock price example     
    """
    args = workload_parser.parse_args()

    api_key , api_secret = acquire_credentials(args.config)

    client = StockHistoricalDataClient(api_key=api_key,
                                       secret_key=api_secret)

    request_params = StockBarsRequest(
        symbol_or_symbols="IBM",
        timeframe=TimeFrame.Minute,
        start=datetime.strptime("2024-05-24", '%Y-%m-%d')
    )

    bars = client.get_stock_bars(request_params)
    df = bars.df 
    
    closing_prices = df['close'].values

    pd.DataFrame(closing_prices, columns=['price']).to_csv('ibm_prices.csv', index=False)


if __name__ == '__main__':
    main()