import pandas as pd

from environment import TradingEnv, train_agent, test_agent

lookback_window_size = 50
data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
train_size = int(len(df) * 0.7)

train_df = df[:train_size]
test_df = df[train_size:]

lookback_window_size = 50

train_env = TradingEnv(train_df, lookback_window_size=lookback_window_size)
test_env = TradingEnv(test_df, lookback_window_size=lookback_window_size)

train_agent(train_env, visualize=False, train_episodes=1, training_batch_size=500)
test_agent(test_env, visualize=True, test_episodes=1)
#Random_games(test_env, visualize=False, train_episodes = 1000)