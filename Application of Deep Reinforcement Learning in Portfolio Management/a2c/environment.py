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


from model import Actor, Critic
from utils import TradingGraph, RolloutBuffer

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TradingEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, render_range=100):
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1 # 따라서 trading 데이터를 df에 넣어줘야 한다
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.render_range = render_range

        self.buffer = RolloutBuffer()

        # Neural networks part bellow
        self.lr = 0.0001
        self.epochs = 1
        self.normalise_value = 100000       
        self.batch_size = 64 
        self.gamma = 0.99
        self.n_epochs = 10
        self.lmbda = 0.5
        self.clip_ratio = 0.2

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.state_size = (self.lookback_window_size, 10)

        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)

        # Create actor-critic network model
        self.Actor = Actor().to(device)
        self.Critic = Critic().to(device)

        self.actor_optimizer = optim.Adam(self.Actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.Critic.parameters(), lr=self.lr)

    def reset(self, env_steps_size = 0):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        
        self.visualisation = TradingGraph(render_range=self.render_range)
        self.trades = deque(maxlen=self.render_range)

        self.env_steps_size = env_steps_size

        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

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

    def act(self, state, testmode=False):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.Actor(state)
            m = Normal(mu, std)
            if testmode==False:
                action = torch.normal(mean=mu, std=std)
            else:
                action = mu
            log_prob = m.log_prob(action)

        #action = torch.clamp(action, -1.0, 1.0)
        action = action.cpu()
        log_prob = log_prob.cpu()
        return action, log_prob

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

#        action = torch.clamp(action, -1.0, 1.0)
        #action = action.cpu()
        
        if action == 0: # hold
            pass

        elif action > 0 and self.balance > self.initial_balance/100: # buy
            action = min(action, 1)
            self.crypto_bought = self.balance * action / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': date, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': 'buy' })
        
        elif action < 0 and self.crypto_held > 0: # sell
            action = max(action, -1)
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

        obs = self._next_observation() #/ self.normalise_value


        return obs, reward, done
    
    def render(self, visualise=False):
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

            self.visualisation.render(date, open, high, low, close, volume, self.net_worth, self.trades)

#    def get_gaes(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.95, normalize=False):
#        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
#        deltas = np.stack(deltas)
#        gaes = copy.deepcopy(deltas)
#        for t in reversed(range(len(deltas) - 1)):
#            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]
#
#        target = gaes + values
#        #if normalize:
#        #    gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
#        return np.vstack(gaes), np.vstack(target)

#    def replay(self, states, actions, rewards, predictions, dones, next_states):
#        # reshape memory to appropriate shape for training
#        states = np.vstack(states)
#        next_states = np.vstack(next_states)
#        actions = np.vstack(actions)
#        predictions = np.vstack(predictions)
#
#        # Compute discounted rewards
#        #discounted_r = np.vstack(self.discount_rewards(rewards))
#
#        # Get Critic network predictions 
#        values = self.critic.predict(states)
#        next_values = self.critic.predict(next_states)
#        
#        # Compute advantages
#        #advantages = discounted_r - values
#        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
#
#        # stack everything to numpy array
#        y_true = np.hstack([advantages, actions])
#        
#        # training Actor and Critic networks
#        a_loss = self.actor.actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=True)
#        c_loss = self.critic.critic.fit(states, target, epochs=self.epochs, verbose=0, shuffle=True)
    

    def optimize_model(self, states, actions, rewards, dones, next_states, log_probs):
        # 현재 T 내의 모든 state, action, reward, done, next_state 확보했음
        # training_batch_size = len(states)
        returns = [0 for _ in range(len(states))]
        returns[0] = rewards[0]
        for i in range(1, len(states)):
            returns[i] += rewards[i] + returns[i-1]

        deltas = [0 for _ in range(len(states)-1)]
        advantages = [0 for _ in range(len(states)-1)]
        self.Actor.train() # training 모드로 변경
        self.Critic.train() # training 모드로 변경

        states = torch.from_numpy(np.array(states)).to(torch.float32).to(device)
        next_states = torch.from_numpy(np.array(next_states)).to(torch.float32).to(device)
        actions = torch.from_numpy(np.array(actions)).to(torch.float32).to(device)
        log_probs = torch.from_numpy(np.array(log_probs[:-1])).to(torch.float32).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        advantages[-1] = rewards[-1]

        # 1-step TD error
        for t in reversed(range(len(deltas)-1)):
            advantages[t] = rewards[t] + self.gamma * self.Critic(next_states[t]) - self.Critic(states[t])
        
        #gaes.append(0)
        
        

        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)

        actor_loss = torch.mul(advantages, log_probs).mean().requires_grad_()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        critic_loss = torch.mul(advantages, advantages).mean().requires_grad_()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()




#        dts = TensorDataset(states, actions, returns, advantages, log_probs)
#        loader = DataLoader(dts, batch_size=len(deltas), shuffle=False)
#
#
#        state, action, ret, advantage, log_prob = batch
#        actor_loss = (advantage * log_prob).
#        critic_loss = 
#        
#        
#
#        for _ in range(self.n_epochs):
#            critic_losses, actor_losses, entropy_bonuses = [], [], []
#            for batch in loader:
#                state, action, ret, gae, log_prob = batch
#                value = self.Critic(state)
#                critic_loss = F.mse_loss(value, ret)
#                self.critic_optimizer.zero_grad()
#                critic_loss.backward()
#                self.critic_optimizer.step()
#
#                mu, std = self.Actor(state)
#                m = Normal(mu, std)
#                action = torch.normal(mean=mu, std=std)
#                log_prob = m.log_prob(action)
#
#                ratio = (log_prob - log_prob_old).exp()
#                surr1 = gae * ratio
#                surr2 = gae * torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio)
#
#                actor_loss = -torch.min(surr1, surr2).mean()
#                entropy_bonus = -m.entropy().mean()
#                self.actor_optimizer.zero_grad()
#                actor_loss.backward()
#                self.actor_optimizer.step()
#
#                critic_losses.append(critic_loss.item())
#                actor_losses.append(actor_loss.item())
#                entropy_bonuses.append(entropy_bonus.item())
#        
#        result = {'actor_loss': np.mean(actor_losses),
#                  'critic_loss': np.mean(critic_losses),
#                  'entropy_bonus': np.mean(entropy_bonuses)}
#        
#        return result
        

    def save(self, name="ppo"):
        torch.save(self.Actor.state_dict(), f'./a2c/{name}_Actor.h5')
        torch.save(self.Critic.state_dict(), f'./a2c/{name}_Critic.h5')

    def load(self, name="ppo"):
        self.Actor.load_state_dict(torch.load(f'./a2c/{name}_Actor.h5', weights_only=True))
        self.Critic.load_state_dict(torch.load(f'./a2c/{name}_Critic.h5', weights_only=True))


def train_agent(env, visualize=False, train_episodes=20, training_batch_size=500):
    #memory = Memory(training_batch_size)

    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Create episode minibatch
        states, actions, rewards, dones, next_states, log_probs = [], [], [], [], [], []
        for _ in range(training_batch_size):
            env.render(visualize)
            #states.append(np.expand_dims(state, axis=0))
            # Selcet action
            #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            action, log_prob = env.act(state, testmode=False) 
            # Observe next state, reward and done signal
            next_state, reward, done = env.step(action)
            if done:
                break
            
            #memory.push(state, action, next_state, reward)
            # Store (next_state, action, reward) in the episode minibatch
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            #next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob)
            
            # Update state
            state = next_state
        
        # Timestep 내의 모든 state, action, reward, done, next_state 확보했음
        # 확보한 애들을 optimise에 던져줘야 함
        env.optimize_model(states, actions, rewards, dones, next_states, log_probs)
        print(f"Episode {episode} net_worth: {env.net_worth}")
        if episode == train_episodes - 1:
            env.save()


def test_agent(env, visualize=False, test_episodes=10):
    env.load() # load the model
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, log_prob = env.act(state, testmode=True)
            next_state, reward, done = env.step(action)
            print(f"Episode {episode} net_worth: {env.net_worth}")
            if env.current_step == env.end_step:
                #average_net_worth += env.net_worth
                break
            
    #print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))
