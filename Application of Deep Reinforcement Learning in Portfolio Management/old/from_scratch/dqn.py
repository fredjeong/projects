import numpy as np
import pandas as pd
#import torch
#from torch import nn
#from torch.utils.data import TensorDataset, DataLoader
import gymnasium as gym
from environment import DiscreteEnv
from stable_baselines3 import DQN
import matplotlib.pyplot as plt

#from collections import deque # 리스트를 회전시킨다
#from network import Model
#import random

data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
prices = df['close'].values
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]

train_env = DiscreteEnv(train_prices)
model = DQN('MlpPolicy', train_env, verbose=1)#, buffer_size=1000, target_update_interval=100)
model.learn(total_timesteps=len(train_prices)-1, progress_bar=True)

#model.save("./from_scratch/dqn/dqn")

test_prices = prices[train_size:]
test_env = DiscreteEnv(test_prices)
obs, _ = test_env.reset()

total_profit = 0 # 총 수익금액
rewards = [] # 평균 수익률
valuation = []

for _ in range(len(test_prices) - 1):
    action, _ = model.predict(obs)#, deterministic=True)
    #profit = test_env.step(action)
    obs, reward, terminated, truncated, info = test_env.step(action)
    #profit.append(reward_profit[1])
    rewards.append(reward)
    valuation.append(test_env.valuation())

    if terminated or truncated:
        break

# 5. 누적 보상 그래프 그리기
    
print(f"Average rate: {round(sum(rewards)/len(test_prices), 2)}, Total profit: {round(total_profit, 3)}, Greatest_loss: {round(min(rewards), 2)}")

fig = plt.figure(figsize=(10, 6))
plt.plot(valuation, label='Cumulative Profit')
plt.xlabel('Timestep')
plt.ylabel('Profit')
plt.title('Profit Over Time')
plt.legend()
plt.show()
fig.savefig('dqn-profit.png', dpi=fig.dpi)

# python3 ./from_scratch/dqn.py





#env = gym.make(train_prices)
#n_features = len(env.observation_space.high)
#n_actions = env.action_space.n
#
#memory = deque(maxlen = memory_len)
#criterion = nn.MSELoss()
#policy_net = Model(n_features, n_actions)
#target_net = Model(n_features, n_actions)
#target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()
#
#def get_states_tensor(sample, states_idx):
#    sample_len = len(sample)
#    states_tensor = torch.empty(sample_len, n_features, dtype=torch.float32, requires_grad=False)
#
#    for i in range(sample_len):
#        for j in range(n_features):
#            states_tensor[i, j] = sample[i][states_idx][j].item()
#
#    return states_tensor
#
#def get_action(state, e=min_epsilon):
#    if random.random() < e:
#        # explore
#        action = random.randrange(0, n_actions)
#    else:
#        state = torch.tensor(state, dtype = torch.float32)
#        action = policy_net(state).argmax().item()
#
#    return action
#
#def fit(model, inputs, labels):
#    inputs = inputs
#    labels = labels
#    train_ds = TensorDataset(inputs, labels)
#
#    optimizer = torch.optim.Adam(params=model.parameters(), lr = learning_rate)
#    model.train()
#    total_loss = 0.0
#
#    for x, y in train_prices:
#        pass
#
#
#    model.eval()
#
#    return total_loss / len(inputs)
#
#def optimize_model(train_batch_size):
#    pass
#
#def train_one_episode():
#    pass
#
#def test():
#    pass
#
#def main(): # 스크립트가 직접 실행될 때만 main()함수를 호출, 이 스크립트를 모듈로 사용할 때는 main()함수가 실행되지 않아 의도하지 않은 동작을 방지할 수 있다.
#
#