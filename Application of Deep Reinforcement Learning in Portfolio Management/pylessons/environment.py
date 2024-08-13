import numpy as np
import pandas as pd
import random
from collections import deque

from utils import TradingGraph


class DiscreteEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, render_range=100):
        '''
        The environment expects a pandas dataframe to be passed that contains the market data to be learned from. 
        Adding to this, we must know our dataset length, starting trading balance, 
        and how many steps of market memory we want our agent to "see", 
        we define all of these parameters in our __init__ part.

        I describe them as a deque list; this means that our list has a limited size of 50 steps. 
        When we append a new item into the list, the last one is removed. 
        '''
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1 # 따라서 trading 데이터를 df에 넣어줘야 한다
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size

        self.action_space = np.array([0, 1, 2]) # 0: hold, 1: buy, 2: sell

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history cnotains the Open, High, Low, Close values as well as volumes.
        # 볼륨은 혼자서 성격이 다르니까 필요하면 뺄수도?
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market + Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10) # OHLC 가격, 012 액션, 현금, 총자산

        self.render_range = render_range # render range in visualisation

    def reset(self, env_steps_size = 0):
        '''
        The reset method should be called every time a new environment is created or to reset an existing envrionment's state.
        '''
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        '''
        훈련 모드에서는 훈련 기간 내 임의의 한 시점부터 거래를 시작하고, 실전 모드에서는 50일차부터 거래를 시작한다. 50일간의 정보는 가지고 있어야 거래를 할 수 있으므로
        '''
        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        self.visualisation = TradingGraph(render_range=self.render_range)
        self.trades = deque(maxlen=self.render_range) # limited orders memory for visualisation

        # 50일간의 데이터를 추가한다.
        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'open'],
                            self.df.loc[current_step, 'high'],
                            self.df.loc[current_step, 'low'],
                            self.df.loc[current_step, 'close'],
                            self.df.loc[current_step, 'volume']
                            ])
        state = np.concatenate((self.market_history, self.orders_history), axis = 1)
        return state

    def step(self, action):
        '''
        Our agent will choose and take either buy, sell, or hold action, calculate the reward, and return the next observation at each step. 
        In a real situation, usually, our price fluctuates in a 1h timeframe (I chose 1 hour in this tutorial) up and down until it closes. 
        In historical data, we can't see these movements, so we need to create them. 
        I do this by taking a random price between open and close prices. 
        Because I am using my total balance, I can easily calculate how much Bitcoin amount I will buy/sell 
        and represent that in balance, crypto_bought, crypto_held, crypto_sold, and net_worth parameters (I add them to my orders_history, 
        so I will send these values to my agent). 
        Also, I can calculate rewards by subtracting net worth from the previous step and current step. 
        And lastly, we must pick a new state by calling _next_observation().
        '''
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        date = self.df.loc[self.current_step, 'date_open']
        high = self.df.loc[self.current_step, 'high']
        low = self.df.loc[self.current_step, 'low']

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'open'],
            self.df.loc[self.current_step, 'close'])
        
        if action == 0: # hold
            pass
        elif action == 1 and self.balance > 0:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price # 거래비용은 무시한다 일단
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': date, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': 'buy' })
        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date': date, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': 'sell' })

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        reward = self.net_worth - self.prev_net_worth

        # 원금의 50% 이상 잃으면 거래를 멈춘다.
        if self.net_worth <= self.initial_balance / 2: 
            done = True
        else:
            done = False

        obs = self._next_observation()

        #write_to_file(date, self.orders_history[-1])

        return obs, reward, done
    
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'open'],
                                    self.df.loc[self.current_step, 'high'],
                                    self.df.loc[self.current_step, 'low'],
                                    self.df.loc[self.current_step, 'close'],
                                    self.df.loc[self.current_step, 'volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    def render(self, visualise = True):
        '''
        Usually, we want to see how our agent learns, performs, etc. So we need to create a render function. 
        For simplicity's sake, we will render the current step of our environment and the net worth so far.
        '''
        #print(f"Step: {self.current_step}, Net worth: {self.net_worth}")
        if visualise:
            date = self.df.loc[self.current_step, 'date_open']
            open = self.df.loc[self.current_step, 'open']
            close = self.df.loc[self.current_step, 'close']
            high = self.df.loc[self.current_step, 'high']
            low = self.df.loc[self.current_step, 'low']
            volume = self.df.loc[self.current_step, 'volume']

            # Render the envrionment to the screen
            self.visualisation.render(date, open, high, low, close, volume, self.net_worth, self.trades)




# 시뮬레이션을 위한 
def Random_games(env, train_episodes = 50, training_batch_size=500):
    average_net_worth = 0
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        while True:
            env.render()

            action = np.random.randint(3, size=1)[0]

            state, reward, done = env.step(action)

            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", env.net_worth)
                break

    print("average_net_worth:", average_net_worth/train_episodes)



data_path = "./data/binance-BTCUSDT-1h.pkl"
df = pd.read_pickle(data_path)
# int(0.7 * len(prices)) = 26016.000
train_size = int(len(df) * 0.7)
train_df = df[:train_size]
test_df = df[train_size:]

lookback_window_size = 50

train_env = DiscreteEnv(train_df, lookback_window_size=lookback_window_size)
test_env = DiscreteEnv(test_df, lookback_window_size=lookback_window_size)

Random_games(train_env, train_episodes = 10, training_batch_size=500)

'''
Usually, a professional trader would most likely look at some charts of price action, 
perhaps overlaid with a couple of technical indicators. 
From there, they would combine this visual information with their prior knowledge of similar price actions 
to make an informed decision in what direction the price is likely to move.

So, we need to translate these human actions into code so that our custom-created agent can understand price action similarly. 
We want state_size to contain all of the input variables that we need our agent to consider before taking action. 
I want that my agent could "see" the main market data points (open price, high, low, close, and daily volume) 
for the last 50 days, as well as a couple of other data points like its account balance, current open positions, and current profit.
'''


'''
Reward:
We want to promote long-term profit to calculate the account balance difference between the previous step and the current step for each step. 
We want our agents to maintain a higher balance for longer, rather than those who rapidly gain money using unsustainable strategies.
'''