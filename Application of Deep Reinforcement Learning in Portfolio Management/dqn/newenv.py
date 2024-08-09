import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import pickle
import matplotlib.pyplot as plt

# 1 coin per trade
HMAX_NORMALIZE = 100 

# Initial amount of money we have in our account
INITIAL_ACCOUNT_BALANCE = 20000

# Total number of cryptocurrencies in our portfolio
STOCK_DIM = 1 # Bitcoin

# Transaction fee: 0.15%
TRANSACTION_FEE_PERCENT = 0.0015
REWARD_SCALING = 1e-4 # what is this?

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day = 0):
        self.df = df
        self.day = day # day라고 썼지만 사실은 time이다.
        time = self.df.iloc[self.day, :]


        # Action space normalization and shape is STOCK_DIM
        self.action_space = spaces.Box(low = -1, high = 1, shape = (STOCK_DIM,)) # 이거 discrete하게 바꿔야함
        # shape = [current balance] + [prices] + [owned shares]

        self.observation_space = spaces.Box(low = 0, high = np.inf, shape = (3,))

        # load data from a pandas dataframe
        self.data = self.df.loc[time,:] # 이부분 수정필요
        self.terminal = False
        """
        df에서 iloc으로 시간 지정 어떻게 할지 생각
        """

        # Initialise state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
            self.data.feature_close.values.tolist() + \
            [0]*STOCK_DIM

        # Initialise reward
        self.reward = 0
        self.cost = 0
        # Memorise all the total balance cahnge
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []
        self.trades = 0
        self._seed()

    def _sell(self, index, action):
        # 내 경우에는 index가 필요없지 않나? 그냥 index = 0으로 두면 됨
        # perform sell action based on the sign of the action
        if self.state[index + STOCK_DIM + 1] > 0: # if my shares are greater than 0:
            # update balance
            self.state[0] += \
                self.state[index+1] * \
                min(abs(action), self.state[index+STOCK_DIM+1]) * \
                (1 - TRANSACTION_FEE_PERCENT)

            self.state[index+STOCK_DIM+1] -= min(abs(action), self.state[index+STOCK_DIM+1])
            self.cost += \
                self.state[index+1]*min(abs(action), self.state[index+STOCK_DIM+1]) * \
                TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _buy(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index+1] 

        # update balance
        self.state[0] -= \
            self.state[index+1] * min(available_amount, action) * \
            (1 + TRANSACTION_FEE_PERCENT)

        self.trades += 1

    def step(self, action):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            plt.plot(self.asset_memory,'r')
            plt.savefig('results/account_value_train.png')
            plt.close()
            end_total_asset = \
                self.state[0] + \
                sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):])) 
                # price * holdings
                # 종목 3개 있으면 (보유현금, 가격1, 가격2, 가격3, 보유량1, 보유량2, 보유량3)이니까 3:5
            
            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('results/account_value_train.csv')
            df_total_value.columns = ['account_value']
            df_total_value['daily_return']=df_total_value.pct_change(24)
            sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / df_total_value['daily_return'].std()
            df_rewards = pd.DataFrame(self.rewards_memory)
            return self.state, self.reward, self.terminal, {}
        else:
            actions = actions * HMAX_NORMALIZE
            
            begin_total_asset = self.state[0] + \
                sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):]))
            
            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell(index, actions[index])

            for index in buy_index:
                self._buy(index, actions[index])
            
            self.day += 1
            self.data = self.df.loc[self.day,:]

            self.state = \
                [self.state[0]] + \
                self.data.adjcp.values.tolist() + \
                list(self.state[(STOCK_DIM+1):])
            
            end_total_asset = \
                self.state[0] + \
                sum(np.array(self.state[1:(STOCK_DIM+1)]) * np.array(self.state[(STOCK_DIM+1):]))
            
            self.asset_memory.append(end_total_asset)
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)

            self.reward = self.reward + REWARD_SCALING

        return self.state, self.reward, self.terminal, {}
            


    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day,:]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = \
            [INITIAL_ACCOUNT_BALANCE] + \
            self.data.adjcp.values.tolist() + \
            [0]*STOCK_DIM
        return self.state

    def render(self, mode='human'):
        return self.state
    
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]