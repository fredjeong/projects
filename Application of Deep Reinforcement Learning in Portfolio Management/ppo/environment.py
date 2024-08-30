import numpy as np
import pandas as pd
import random
from collections import deque
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, Dataset

import gymnasium as gym
from tensorboardX import SummaryWriter


from model import Actor, Critic
from utils import TradingGraph, RolloutBuffer

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TradingEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=1, render_range=100):
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.render_range = render_range

        self.memory = RolloutBuffer()

        self.lr = 0.0003
        #self.epochs = 1     
        self.batch_size = 64 
        self.n_epochs = 10
        self.lmbda = 0.5
        self.clip_ratio = 0.2 
        self.vf_coef = 1.0
        self.ent_coef = 0.01

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state_size = (self.lookback_window_size, 10)

        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.Actor = Actor().to(device)
        self.Critic = Critic().to(device)

        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr=self.lr)
        
        self.policy_losses = []
        self.value_losses = []
        self.portfolio_values = []
        self.benchmark = []
    

    def reset(self, env_steps_size = 0):
        self.portfolio_values = []
        self.benchmark = []
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        
        self.visualisation = TradingGraph(render_range=self.render_range)
        self.trades = deque(maxlen=self.render_range)

        self.env_steps_size = env_steps_size

        if env_steps_size > 0: # used for training
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for test
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        # Add the market history of the past 50 time steps which help the agent making decision
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

    def act(self, state, testmode):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.Actor(state)
            m = Normal(mu, std)
            action = torch.normal(mean=mu, std=std)
            if testmode==False:
                action = torch.normal(mean=mu, std=std)
            else:
                action = mu
            log_prob_old = m.log_prob(action)

        #action = torch.clamp(action, -1.0, 1.0)
        action = torch.tanh(action)
        log_prob_old = log_prob_old.cpu()

        return action.cpu(), log_prob_old

    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'open'],
                                    self.df.loc[self.current_step, 'high'],
                                    self.df.loc[self.current_step, 'low'],
                                    self.df.loc[self.current_step, 'close'],
                                    self.df.loc[self.current_step, 'volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
        return obs

    def step(self, action):
        action = action.item()
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

        elif action > 0 and self.balance > self.initial_balance/100: # buy
            self.crypto_bought = self.balance * action / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': date, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': 'buy' })
        
        elif action < 0 and self.crypto_held > 0: # sell
            self.crypto_sold = self.crypto_held * abs(action) 
            self.balance += self.crypto_sold * abs(action) * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date': date, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': 'sell' })

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance / 2: 
            done = True
        else:
            done = False

        obs = self._next_observation() 

        return obs, reward, done
    
    def render(self, visualise=False):
        if visualise:
            date = self.df.loc[self.current_step, 'date_open']
            open = self.df.loc[self.current_step, 'open']
            close = self.df.loc[self.current_step, 'close']
            high = self.df.loc[self.current_step, 'high']
            low = self.df.loc[self.current_step, 'low']
            volume = self.df.loc[self.current_step, 'volume']

            self.visualisation.render(date, open, high, low, close, volume, self.net_worth, self.trades)

    def optimize_model(self, states, actions, rewards, dones, next_states, log_prob_olds):
        if len(states) < self.batch_size:
            return
        returns = [0 for _ in range(len(states))]
        returns[0] = rewards[0]
        for i in range(1, len(states)):
            returns[i] += rewards[i] + returns[i-1]

        self.Actor.train() # training 모드로 변경
        self.Critic.train() # training 모드로 변경

        states = torch.from_numpy(np.array(states)).to(torch.float32).to(device)
        next_states = torch.from_numpy(np.array(next_states)).to(torch.float32).to(device)
        actions = torch.from_numpy(np.array(actions)).to(torch.float32).to(device)
        log_prob_olds = torch.from_numpy(np.array(log_prob_olds)).to(torch.float32).to(device)
        rewards = torch.from_numpy(np.array(rewards)).to(torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        with torch.no_grad():
            deltas = rewards + self.Critic(next_states) - self.Critic(states)
            gaes = torch.clone(deltas)
            for t in reversed(range(len(deltas)-1)):
                gaes[t] = deltas[t] + deltas[t+1] * self.lmbda

        
        dts = TensorDataset(states, actions, returns, gaes, log_prob_olds)
        loader = DataLoader(dts, batch_size=self.batch_size, shuffle=True, drop_last=True)

        counter = 0
        for e in range(self.n_epochs):
            for batch in loader:
                s_, a_, ret_, gae_, log_prob_old_ = batch
                value = self.Critic(s_).squeeze()
                value_loss = F.mse_loss(value, ret_)

                mu, std = self.Actor(s_)
                d = Normal(mu, std)
                z = torch.atanh(torch.clamp(a_, -1.0 + 1e-7, 1.0 - 1e-7))
                log_prob = d.log_prob(z).sum(dim=-1, keepdims=True)

                ratio = (log_prob - log_prob_old_).exp()
                surr1 = gae_ * ratio
                surr2 = gae_ * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)

                policy_loss = -torch.min(surr1, surr2).mean()
                entropy_bonus = -d.entropy().mean()

                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_bonus
                self.critic_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                loss.backward()
                self.critic_optimizer.step()
                self.actor_optimizer.step()
                counter+=1

                self.value_losses.append(value_loss.item())
                self.policy_losses.append(policy_loss.item())

    def save(self, name="ppo"):
        torch.save(self.Actor.state_dict(), f'./ppo/{name}_Actor.h5')
        torch.save(self.Critic.state_dict(), f'./ppo/{name}_Critic.h5')

    def load(self, name="ppo"):
        self.Actor.load_state_dict(torch.load(f'./ppo/{name}_Actor.h5', weights_only=False))
        self.Critic.load_state_dict(torch.load(f'./ppo/{name}_Critic.h5', weights_only=False))


def train_agent(env, train_episodes, training_batch_size, visualize=False):
    total_average = deque(maxlen=env.n_epochs)
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        
        states, actions, rewards, dones, next_states, log_prob_olds = [], [], [], [], [], []
        for _ in range(training_batch_size):
            env.render(visualize)
            action, log_prob_old = env.act(state, testmode=False) 
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_prob_olds.append(log_prob_old)
            if done:
                break
            state = next_state
            env.portfolio_values.append(env.net_worth)
            env.benchmark.append(env.df.loc[env.current_step, 'close'] * env.initial_balance / env.df.loc[env.start_step, 'close'])
            
        env.optimize_model(states, actions, rewards, dones, next_states, log_prob_olds)
        total_average.append(env.net_worth)

        print(f"Episode {episode} net_worth: {env.net_worth}")
        np.savetxt(f'./ppo/records/benchmark_{episode}.csv', env.benchmark)
        np.savetxt(f'./ppo/records/portfolio_values_{episode}.csv', env.portfolio_values)
        if episode == train_episodes - 1:
            env.save()

    np.savetxt('./ppo/records/policy_losses.csv', env.policy_losses)
    np.savetxt('./ppo/records/value_losses.csv', env.value_losses)


def test_agent(env, test_episodes, visualize=False):
    env.load() 
    for episode in range(test_episodes):
        state = env.reset()
        states, actions, rewards, dones, next_states, log_prob_olds = [], [], [], [], [], []
        while True:
            env.render(visualize)
            action, log_prob_old = env.act(state, testmode=True)
            next_state, reward, done = env.step(action)
            #states.append(np.expand_dims(state, axis=0))
            #next_states.append(np.expand_dims(next_state, axis=0))
            #actions.append(action)
            #rewards.append(reward)
            #dones.append(done)
            #log_prob_olds.append(log_prob_old)
            state = next_state
            #env.optimize_model(states, actions, rewards, dones, next_states, log_prob_olds)
            #print(f"Episode {episode} Step {env.current_step} net_worth: {env.net_worth}")
            env.portfolio_values.append(env.net_worth)
            if env.current_step == env.end_step:
                break
    np.savetxt('./ppo/portfolio_values.csv', env.portfolio_values)