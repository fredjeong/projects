import pandas as pd

from environment import TradingEnv, train_agent

lookback_window_size = 50
data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)

train_size = int(len(df) * 0.7)
train_df = df[:train_size]

train_env = TradingEnv(df, lookback_window_size=lookback_window_size)

train_agent(train_env, visualize=False, train_episodes=20, training_batch_size=3000)

#2260