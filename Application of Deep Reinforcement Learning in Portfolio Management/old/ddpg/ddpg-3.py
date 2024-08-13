import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import random

# 난수 시드 설정
random_seed = random.randint(0, 1000000)
np.random.seed(random_seed)

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
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,), dtype=np.float32)
        self.done = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.fiat = self.initial_fiat
        self.asset = 0
        self.done = False
        self.truncated = False
        return self._get_observation(), {}

    def step(self, action):
        previous_value = self._get_value()

        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            self.truncated = True
        
        current_price = self.prices[self.current_step]
        
        action = action[0]  # DDPG returns a 2D array, we need to extract the single value
        if action > 0:  # Buy
            buy_amount = min(self.fiat, self.fiat * action)
            self.asset += buy_amount * (1 - self.transaction_cost) / current_price
            self.fiat -= buy_amount
        elif action < 0:  # Sell
            sell_amount = min(self.asset, self.asset * abs(action))
            self.fiat += sell_amount * current_price * (1 - self.transaction_cost)
            self.asset -= sell_amount

        current_value = self._get_value()
        reward = (current_value - previous_value) / previous_value
        reward *= 100  # Scale up the reward

        if current_value <= 0:
            self.done = True
            reward = -100  # Big penalty for bankruptcy

        print(f"Step: {self.current_step}, Action: {action}, Reward: {reward}, Value: {current_value}")  # Logging

        return self._get_observation(), reward, self.done, self.truncated, {}

    def _get_observation(self):
        return np.array([self.fiat, self.asset, self.prices[self.current_step]], dtype=np.float32)

    def _get_value(self):
        return self.fiat + self.asset * self.prices[self.current_step]

# 3. 환경 생성 및 모델 학습
train_env = DummyVecEnv([lambda: BitcoinTradingEnv(train_prices)])
test_env = DummyVecEnv([lambda: BitcoinTradingEnv(test_prices)])

n_actions = train_env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = DDPG('MlpPolicy', train_env, action_noise=action_noise, verbose=1,
             learning_rate=1e-4,
             buffer_size=100000,
             learning_starts=1000,
             batch_size=64,
             gamma=0.99)

model.learn(total_timesteps=200000)  # Increased learning steps

# 4. 모델 저장 및 평가
model.save("./models/ddpg_model")
obs = test_env.reset()

cumulative_rewards = []
total_reward = 0

for _ in range(len(test_prices) - 1):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    
    total_reward += reward[0]
    cumulative_rewards.append(total_reward)
    
    if done:
        break

# 5. 누적 보상 그래프 그리기
fig = plt.figure(figsize=(10, 6))
plt.plot(cumulative_rewards, label='Cumulative Reward')
plt.xlabel('Time Step')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward Over Time')
plt.legend()
plt.show()
fig.savefig('ddpg-cumulative-rewards.png', dpi=fig.dpi)

print(f"Total Reward: {total_reward}")
print(f"Random Seed: {random_seed}")