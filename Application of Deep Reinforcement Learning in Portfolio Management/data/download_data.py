from gym_trading_env.downloader import download
import datetime
import pandas as pd

download(exchange_names = ["binance"],
         symbols = ["BTC/USDT"],
         timeframe = "1h",
         dir = "data",
         since = datetime.datetime(year = 2020, month = 3, day = 31))

# To load the data
# df = pd.read_pickle("./data/binance-BTCUSDT-1h.pkl")

