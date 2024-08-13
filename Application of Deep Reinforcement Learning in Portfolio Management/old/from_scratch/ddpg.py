import numpy as np
import pandas as pd
#import torch
#from torch import nn
#from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
from environment import ContinuousEnv
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt

#from collections import deque # 리스트를 회전시킨다
#from network import Model
#import random

data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
prices = df['close'].values
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]

train_env = ContinuousEnv(train_prices)
model = DDPG('MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=len(train_prices)-1, progress_bar=True)

#model.save("./from_scratch/dqn/dqn")

test_prices = prices[train_size:]
test_env = ContinuousEnv(test_prices)
obs, _ = test_env.reset()

total_profit = 0 # 총 수익금액
rewards = [] # 평균 수익률
cumulative_profit = []

for _ in range(len(test_prices) - 1):
    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = test_env.step(action)
    #profit.append(reward_profit[1])
    rewards.append(reward)
    total_profit += reward * test_env.prices[test_env.idx - 1] / 100
    cumulative_profit.append(total_profit)

    if terminated or truncated:
        break

# 5. 누적 보상 그래프 그리기
    
print(f"Average rate: {round(sum(rewards)/len(test_prices), 3)}, Total profit: {round(total_profit, 3)}, Greatest_loss: {round(min(rewards), 3)}")

fig = plt.figure(figsize=(10, 6))
plt.plot(cumulative_profit, label='Cumulative Profit')
plt.xlabel('Timestep')
plt.ylabel('Profit')
plt.title('Profit Over Time')
plt.legend()
plt.show()
fig.savefig('ddpg-profit.png', dpi=fig.dpi)

# python3 ./from_scratch/ddpg.py