from pprint import pprint
import yaml 
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest
import argparse 


def get_keys() -> None:
    """
    opens keys from paca yaml 
    """
    with open("./.paca-token.yml",'r') as f: 
        key_information = yaml.safe_load(f)

    return key_information['api_key'], key_information['api_secret']


def create_vanilla_order(ticker: str, num: int) -> MarketOrderRequest: 
    """
    creates a very simple order for a buy in the market. 
    """

    request = MarketOrderRequest(symbol = ticker, 
                                 qty=num,
                                 side=OrderSide.BUY,
                                 time_in_force=TimeInForce.DAY)

    return request 

def submit_order(client: TradingClient, order_request):
    """
    simple function to submit order. 
    """
    client.submit_order(order_data=order_request)

async def trade_stats(data):
    print(data) 

def main(): 
    """
    
    """

    # basic setup
    k, s = get_keys()
    trading_client = TradingClient(api_key=k, secret_key=s, paper=True)

    # print things from the account 
    account = trading_client.get_account()

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--ticker',required=True)
    parser.add_argument('-n', '--number', required=True)

    args = parser.parse_args() 

    ticker = args.ticker
    num = args.number
    # simple order.   
    pprint("Creating order")
    order = create_vanilla_order(ticker=ticker, num=num)
    pprint("submitting")
    submit_order(trading_client,order)
    pprint("submitted")



if __name__ == '__main__':
    main()