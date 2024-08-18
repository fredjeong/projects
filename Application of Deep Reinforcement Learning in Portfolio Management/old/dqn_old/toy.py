import pandas as pd
from environment import TradingEnv, Random_games

lookback_window_size = 50
data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
train_size = int(len(df) * 0.7)

train_df = df[:train_size]
test_df = df[train_size:]

lookback_window_size = 50

train_env = TradingEnv(train_df, lookback_window_size=lookback_window_size)
test_env = TradingEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, visualize=False,train_episodes = 10, training_batch_size=500)
Random_games(test_env, visualize=True, train_episodes = 1, training_batch_size=300)