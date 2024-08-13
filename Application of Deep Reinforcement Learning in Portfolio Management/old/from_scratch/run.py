import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium as gym

import torch
from network import network
from from_scratch.environment import TradingEnv


data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
prices = df['close'].values
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]
#test_prices = prices[train_size:]

train_env = TradingEnv(train_prices)

policy_net = Model(len(), )

terminated = False
truncated = False
while not terminated:
    state = torch.tensor(state, dtype = torch.float32)
    action = torch.argmax(policy_net(state)).item()
    state, reward, terminated, truncated, _ = train_env.step(action)
    




print(f"Total Reward: {total reward}")