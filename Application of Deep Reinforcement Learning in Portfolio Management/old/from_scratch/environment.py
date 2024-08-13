import gymnasium as gym
import numpy as np


class DiscreteEnv(gym.Env):
    def __init__(self, prices, initial_fiat=10000, transaction_cost = 0.001, ratio=0.1):
        super(DiscreteEnv, self).__init__()
        self.initial_fiat = initial_fiat
        self.transaction_cost = transaction_cost
        self.ratio = ratio # 현금/코인의 몇 프로를 한 번에 거래할 것인지
        self.idx = 0 # 현재 타임스텝
        self.fiat = initial_fiat # 현금 보유량
        self.asset = 0 # 코인 보유량
        self.prices = prices # 종가(close price) 정보
        
        self.action_space = gym.spaces.Discrete(3) # 0: 구매, 1: 매도, 2: 홀드
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,)) # observation: fiat, asset, price

        self.terminated = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        '''
        Resets the environment to an initial state, required before calling step.
        Returns the first agent observation for an episode and information, i.e. metrics, debug info
        '''
        self.idx = 0
        self.fiat = self.initial_fiat
        self.asset = 0
        self.terminated = False
        self.truncated = False

        obs = self.get_obs()

        return obs, {}

    def step(self, action):
        '''
        Updates an environment with actions returning:
            - the next agent observation, 
            - the reward for taking that action
            - if the environment has terminated or truncated due to the latest action
            - information from the envrionment about the step i.e., metrics, debug info
        '''
        previous_value = self.valuation()

        self.idx += 1
        if self.idx >= len(self.prices) - 1:
            self.done = True

        p = self.ratio * self.prices[self.idx]
        if action == 0: # 구매
            if self.fiat >= p:
                # 여기 수정해야함
                self.asset += self.fiat * (self.ratio) * (1 - self.transaction_cost) / self.prices[self.idx]
                self.fiat *= (1 - self.ratio)
        elif action == 1: # 매도
            if self.asset >= 0:
                self.fiat += self.asset * (self.ratio) * self.prices[self.idx] * (1 - self.transaction_cost)
                self.asset *- (1 - self.ratio)
        else: # 홀드
            pass
        
        current_value = self.valuation()

        if current_value <= 0:
            self.truncated = True
            reward = -1  # 파산 시 큰 페널티
        else:
            if action == 0: # 매수
                if self.fiat >= p:
                    reward = (self.prices[self.idx] - self.prices[self.idx - 1]) / self.prices[self.idx - 1]
                else:
                    reward = -1

            elif action == 1: # 매도
                reward = (self.prices[self.idx - 1] - self.prices[self.idx]) / self.prices[self.idx - 1]
            else:
                reward = 0
            #reward = current_value/previous_value# 수익률 퍼센테이지로 표현
            #reward = (current_value - previous_value) / current_value
        
        obs = self.get_obs()

        return obs, reward, self.terminated, self.truncated, {}

    def get_obs(self):
        return np.array([self.fiat, self.asset, self.prices[self.idx]])
    
    def valuation(self):
        return self.fiat + self.asset * self.prices[self.idx]
    

class ContinuousEnv(gym.Env):
    def __init__(self, prices, initial_fiat=20000, transaction_cost = 0.001, ratio=0.1):
        super(ContinuousEnv, self).__init__()
        self.initial_fiat = initial_fiat
        self.transaction_cost = transaction_cost
        self.ratio = ratio # 현금/코인의 몇 프로를 한 번에 거래할 것인지
        self.idx = 0 # 현재 타임스텝
        self.fiat = initial_fiat # 현금 보유량
        self.asset = 0 # 코인 보유량
        self.prices = prices # 종가(close price) 정보
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(3,)) # observation: fiat, asset, price

        self.terminated = False
        self.truncated = False

    def reset(self, seed=None, options=None):
        '''
        Resets the environment to an initial state, required before calling step.
        Returns the first agent observation for an episode and information, i.e. metrics, debug info
        '''
        self.idx = 0
        self.fiat = self.initial_fiat
        self.asset = 0
        self.terminated = False
        self.truncated = False

        obs = self.get_obs()

        return obs, {}

    def step(self, action):
        '''
        Updates an environment with actions returning:
            - the next agent observation, 
            - the reward for taking that action
            - if the environment has terminated or truncated due to the latest action
            - information from the envrionment about the step i.e., metrics, debug info
        '''
        previous_value = self.valuation()

        self.idx += 1
        if self.idx >= len(self.prices) - 1:
            self.done = True

        if action[0] > 0: # 매수
            if self.fiat > 0:
                self.asset += self.fiat * action[0] * (1 - self.transaction_cost) / self.prices[self.idx]
                self.fiat *= (1 - action[0])
        elif action[0] < 0: # 매도
            if self.asset > 0:
                self.fiat += self.asset * abs(action[0]) * self.prices[self.idx] * (1 - self.transaction_cost)
                self.asset *= (1 - abs(action[0]))
        else: # 홀드
            pass
        
        current_value = self.valuation()

        if current_value <= 0:
            self.truncated = True
            reward = -100  # 파산 시 큰 페널티
        else:
            #reward = 100 * (current_value - previous_value) / previous_value # 수익률 퍼센테이지로 표현
            reward = 100 * (current_value - previous_value) / current_value # 수익률 퍼센테이지로 표현

            #reward = current_value - previous_value
        
        obs = self.get_obs()

        return obs, reward, self.terminated, self.truncated, {}

    def get_obs(self):
        return np.array([self.fiat, self.asset, self.prices[self.idx]])
    
    def valuation(self):
        return self.fiat + self.asset * self.prices[self.idx]