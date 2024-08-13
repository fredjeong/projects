import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym 
from stable_baselines3 import DQN

# 1. 데이터 로드 및 전처리
data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)

# 필요에 따라 특정 열 선택 또는 전처리 작업 수행
# 예시: 가격 열만 사용
prices = df['close'].values

# train/test 데이터셋으로 나누기 (80% train, 20% test)
train_size = int(len(prices) * 0.8)
train_prices = prices[:train_size]
test_prices = prices[train_size:]

def valuation(fiat, asset, price):
    return sum([asset * price, fiat])

# 2. 환경 설정
class BitcoinTradingEnv(gym.Env):
    # 가격 정보 prices와 초기 자금 initial_fiat(단위: 달러) 입력
    def __init__(self, prices, initial_fiat = 20000, transaction_cost = 0.001):
        super(BitcoinTradingEnv, self).__init__()
        '''
        state(현금, 코인 개수, 가격), action space, observation space, current step 초기화 둥
        '''
        self.initial_fiat = initial_fiat
        self.fiat = self.initial_fiat
        self.asset = 0
        self.transaction_cost = transaction_cost

        self.prices = prices
        self.current_step = 0

        # [action_space] The Space object corresponding to valid actions
        # All valid actions should be contained within the space.
        self.action_space = gym.spaces.Discrete(3) # 0, 1, 2 리턴
        # [observation_space] The Space object corresponding to valid actions
        # All valid observations should be contained within the space.
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,)) # 종가만 반영

        self.done = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        '''
        Resets the environment to an initial state, required before calling step.
        Returns the first agent observation for an episode and information.
        '''
        super().reset(seed=seed)
        self.current_step = 0
        self.fiat = self.initial_fiat
        self.asset = 0
        self.done = False
        return np.array([self.fiat, self.asset, self.prices[self.current_step]]), {}

    def step(self, action): 
        '''
        Updates an environment with actions returning the next agent observation, the reward for taking that actions,
        if the environment has terminated or truncated due to the latest action and information from the envrionment about the step
        '''
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1 or valuation(self.fiat, self.asset, self.prices[self.current_step]) <= 0:
            self.done = True
            self.truncated = True
        
        if action == 0: # 구매
            if self.fiat >= 0:
                self.asset += self.fiat * 0.5 * (1 - self.transaction_cost) / self.prices[self.current_step]
                self.fiat = 0
        elif action == 1: # 매도
            if self.asset >= 0:
                self.fiat = self.asset * 0.5 * self.prices[self.current_step] * (1 - self.transaction_cost)
                self.asset = 0
        elif action == 2: # 홀드
            pass

        current_value = valuation(self.fiat, self.asset, self.prices[self.current_step])
        previous_value = valuation(self.fiat, self.asset, self.prices[self.current_step - 1])

        reward = current_value - previous_value 
        obs = np.array([self.fiat, self.asset, self.prices[self.current_step]])
        return obs, reward, self.done, self.truncated, {}

# Train 환경과 Test 환경 생성
train_env = BitcoinTradingEnv(train_prices)
test_env = BitcoinTradingEnv(test_prices)

# 3. DQN 모델 구현 및 학습
model = DQN('MlpPolicy', train_env, verbose=1)
model.learn(total_timesteps=10000)

# 4. 모델 평가 및 저장
model.save("./models/dqn_model")

# Test 모델 평가
obs, _ = test_env.reset()

cumulative_rewards = []  # 누적 보상을 저장할 리스트
total_reward = 0  # 총 보상 초기화

for _ in range(len(test_prices) - 1):
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, truncated, info = test_env.step(action)
    
    total_reward += reward  # 현재 보상을 총 보상에 더함
    cumulative_rewards.append(total_reward)  # 누적 보상을 리스트에 저장
    
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
fig.savefig('dqn-cumulative-rewards', dpi = fig.dpi)

print(f"Total Reward: {total_reward}")