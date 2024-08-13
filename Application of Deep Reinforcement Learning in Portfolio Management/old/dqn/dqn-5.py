import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import random
from collections import deque

# 데이터 로드
df = pd.read_pickle("data/binance-BTCUSDT-1h.pkl")

# 환경 클래스 정의
class TradingEnv:
    def __init__(self, df):
        self.df = df
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.df.iloc[self.current_step]
        return np.array([
            obs['open'], obs['high'], obs['low'], obs['close'], obs['volume']
        ])
    
    def step(self, action):
        self.current_step += 1
        
        if self.current_step >= len(self.df):
            done = True
        else:
            done = False
        
        obs = self._next_observation()
        
        # 간단한 보상 시스템: 가격이 올랐으면 양의 보상, 내렸으면 음의 보상
        reward = obs[3] - self.df.iloc[self.current_step-1]['close']
        
        return obs, reward, done

# DQN 네트워크 정의
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN 에이전트 클래스 정의
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return np.argmax(act_values.detach().numpy())
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                target = (reward + self.gamma *
                          np.amax(self.model(next_state).detach().numpy()))
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state)
            target_f[0][action] = target
            loss = self.criterion(self.model(state), target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# 학습 실행
env = TradingEnv(df)
state_size = 5
action_size = 3  # 매수, 매도, 홀드
agent = DQNAgent(state_size, action_size)

episodes = 1000
batch_size = 32

for e in range(episodes):
    state = env.reset()
    for time in range(len(df)):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print(f"episode: {e}/{episodes}, score: {time}")
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)

print("학습 완료")