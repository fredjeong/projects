# ChatGPT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
from stable_baselines3 import DQN

# 1. 데이터 로드 및 전처리
data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
prices = df['close'].values
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]
test_prices = prices[train_size:]

def valuation(fiat, asset, price):
    return fiat + asset * price

# 2. 환경 설정
class BitcoinTradingEnv(gym.Env):
    def __init__(self, prices, initial_fiat=20000, transaction_cost=0.001):
        super(BitcoinTradingEnv, self).__init__()
        self.initial_fiat = initial_fiat
        self.fiat = self.initial_fiat
        self.asset = 0
        self.transaction_cost = transaction_cost
        self.prices = prices
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(3) # 0: 구매, 1: 매도, 2: 홀드
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,))
        self.done = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.fiat = self.initial_fiat
        self.asset = 0
        self.done = False
        return np.array([self.fiat, self.asset, self.prices[self.current_step]]), {}

    def step(self, action):
        previous_fiat = self.fiat
        previous_asset = self.asset

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1 or valuation(self.fiat, self.asset, self.prices[self.current_step]) <= 0:
            self.done = True
            self.truncated = True
        
        if action == 0:  # 구매
            if self.fiat > 0:
                self.asset += self.fiat * (1 - self.transaction_cost) / self.prices[self.current_step]
                self.fiat = 0
        elif action == 1:  # 매도
            if self.asset > 0:
                self.fiat += self.asset * self.prices[self.current_step] * (1 - self.transaction_cost)
                self.asset = 0
        elif action == 2:  # 홀드
            pass

        current_value = valuation(self.fiat, self.asset, self.prices[self.current_step])
        previous_value = valuation(previous_fiat, previous_asset, self.prices[self.current_step - 1])

        reward = current_value - previous_value  # 실제 가치 변화를 보상으로 계산
        obs = np.array([self.fiat, self.asset, self.prices[self.current_step]])
        return obs, reward, self.done, self.truncated, {}

# 3. 환경 생성 및 모델 학습
train_env = BitcoinTradingEnv(train_prices)
test_env = BitcoinTradingEnv(test_prices)
model = DQN('MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=10000)

# 4. 모델 저장 및 평가
model.save("./models/dqn_model")
obs, _ = test_env.reset()

cumulative_rewards = []
total_reward = 0

for _ in range(len(test_prices) - 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = test_env.step(action)
    
    total_reward += reward
    cumulative_rewards.append(total_reward)
    
    if done or truncated:
        break

# 5. 누적 보상 그래프 그리기
fig = plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards, label='Cumulative Reward')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Time')
plt.legend()
plt.show()
fig.savefig('dqn-cumulative-rewards.png', dpi=fig.dpi)

print(f"Total Reward: {total_reward}")