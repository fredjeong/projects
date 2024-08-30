import torch
import pandas as pd

from environment import TradingEnv, test_agent

# gpu 설정
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

lookback_window_size = 50
data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
train_size = int(len(df) * 0.7)
test_df = df[train_size:]

test_env = TradingEnv(test_df, lookback_window_size=lookback_window_size)

test_agent(test_env, visualize=False, test_episodes=1)
