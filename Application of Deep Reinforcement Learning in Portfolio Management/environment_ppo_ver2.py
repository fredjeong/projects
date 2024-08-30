import numpy as np
import pandas as pd
import random
from collections import deque, namedtuple
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Normal
from torch.utils.data import TensorDataset, DataLoader, Dataset

import gymnasium as gym


from model import Actor, Critic
from utils import TradingGraph, RolloutBuffer, ReplayMemory

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TradingEnv:
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, render_range=100):
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.render_range = render_range

        self.memory = ReplayMemory(50000)

        self.lr = 0.0001

        self.batch_size = 500 
        self.n_epochs = 10
        self.lmbda = 0.95
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

    def act(self, state, testmode=False):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            mu, std = self.Actor(state)
            m = Normal(mu, std)
            if testmode==False:
                action = torch.normal(mean=mu, std=std)
            else:
                action = mu

        #action = torch.clamp(action, -1.0, 1.0)
        action = torch.tanh(action)

        return action.cpu()

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
            #action = min(action, 1)
            self.crypto_bought = self.balance * action / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': date, 'High': high, 'Low': low, 'Total': self.crypto_bought, 'Type': 'buy' })
        
        elif action < 0 and self.crypto_held > 0: # sell
            #action = max(action, -1)
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
    

    def optimize_model(self):
        self.Actor.train()
        self.Critic.train()

        Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        # 일단은 모두 가져오자
        transitions = self.memory.upload()
        self.memory.clear()
        trajectories = Transition(*zip(*transitions))
        states = torch.cat(trajectories.state)
        actions = torch.cat(trajectories.action)
        next_states = torch.cat(trajectories.next_state)
        rewards = torch.cat(trajectories.reward)

        with torch.no_grad():
            delta = rewards + self.Critic(next_states) - self.Critic(states)
            gae = torch.clone(delta)
            ret = torch.clone(rewards)
            for t in reversed(range(len(rewards) - 1)):
                gae[t] += self.lmbda * gae[t+1]
                ret[t] += ret[t+1]
            mu, std = self.Actor(states)
            d = Normal(mu,std)
            a = torch.atanh(torch.clamp(actions, -1.0 + 1e-7, 1.0-1e-7)).to(device)
            log_prob_old = d.log_prob(a).sum(dim=-1, keepdims=True)
        
        dts = TensorDataset(states, a, ret, gae, log_prob_old)
        loader = DataLoader(dts, batch_size=self.batch_size, shuffle=True)
        
        for e in range(self.n_epochs):
            for batch in loader:
                s_, a_, ret_, gae_, log_prob_old_ = batch
                value = self.Critic(s_)
                value_loss = F.mse_loss(value, ret_)

                mu, std = self.Actor(s_)
                d = Normal(mu, std)
                z = torch.atanh(torch.clamp(a_, -1.0 + 1e-7, 1.0 - 1e-7))
                log_prob = d.log_prob(z).sum(dim=-1, keepdims=True)
                
                ratio = (log_prob - log_prob_old).exp()
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

        """
        코드 실행 안되면 아랫부분 실행할것
        """
#        # 현재 T 내의 모든 state, action, reward, done, next_state 확보했음
#        # training_batch_size = len(states)
#        returns = [0 for _ in range(len(states))]
#        returns[0] = rewards[0]
#        for i in range(1, len(states)):
#            returns[i] += rewards[i] + returns[i-1]
#
#        deltas = [0 for _ in range(len(states)-1)]
#        gaes = [0 for _ in range(len(states)-1)]
#
#        self.Actor.train() # training 모드로 변경
#        self.Critic.train() # training 모드로 변경
#
#        states = torch.from_numpy(np.array(states)).to(torch.float32).to(device)
#        next_states = torch.from_numpy(np.array(next_states)).to(torch.float32).to(device)
#        actions = torch.from_numpy(np.array(actions)).to(torch.float32).to(device)
#        log_prob_olds = torch.from_numpy(np.array(log_prob_olds)).to(torch.float32).to(device)
#        returns = torch.tensor(returns, dtype=torch.float32).to(device)
#
#        deltas[-1] = rewards[-2] + self.Critic(next_states[-2]) - self.Critic(states[-2])
#        gaes[-1] = deltas[-1]
#        for t in reversed(range(len(deltas)-1)):
#            deltas[t] = rewards[t] + self.Critic(next_states[t]) - self.Critic(states[t])
#            gaes[t] = deltas[t] + deltas[t+1] * self.lmbda
#        
#        gaes.append(0)
#        
#        ## batch 뽑기
#        #print(states.type())
#        ##states = torch.from_numpy(np.array(states))
#        #actions = torch.tensor(actions)
#        #returns = torch.tensor(returns)
#        #gaes = torch.tensor(gaes)
#        #log_prob_olds = torch.tensor(log_prob_olds)
#        #states = torch.from_numpy(np.array(states)).to(torch.float32).to(device)
#        #actions = torch.from_numpy(np.array(actions)).to(torch.float32).to(device)
#        #returns = torch.from_numpy(np.array(returns)).to(torch.float32).to(device)
#        gaes = torch.tensor(gaes, dtype=torch.float32).to(device)
#    
#
#        dts = TensorDataset(states, actions, returns, gaes, log_prob_olds)
#        loader = DataLoader(dts, batch_size=self.batch_size, shuffle=True)
#
#        for _ in range(self.n_epochs):
#            critic_losses, actor_losses, entropy_bonuses = [], [], []
#            for batch in loader:
#                state, action, ret, gae, log_prob_old = batch
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
        torch.save(self.Actor.state_dict(), f'./ppo_copy/{name}_Actor.h5')
        torch.save(self.Critic.state_dict(), f'./ppo_copy/{name}_Critic.h5')

    def load(self, name="ppo"):
        self.Actor.load_state_dict(torch.load(f'./ppo_copy/{name}_Actor.h5', weights_only=True))
        self.Critic.load_state_dict(torch.load(f'./ppo_copy/{name}_Critic.h5', weights_only=True))


def train_agent(env, visualize=False, train_episodes=20, training_batch_size=500):
    """
    우선 trajectory를 전부 수집해서 replay memory에 넣는다
    다음으로 GAE를 계산한다
    K번 업데이트 과정을 다음과 같이 진행한다:
        1. PPO Total Loss 계산
        2. Critic과 Actor 네트워크 업데이트
    """
    # n_epoch가 진행되는 동안 old probability는 고정이다
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        # Trajectory 수집해서 replay memory에 넣기
        for t in range(training_batch_size):
            env.render(visualize)
            action = env.act(state, testmode=False)
            next_state, reward, done = env.step(action)
            if done:
                break
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
            env.memory.push(state, action, next_state, reward)
            state = next_state
            
        # Optimize
        env.optimize_model()
        print(f"Episode {episode} net_worth: {env.net_worth}")
        if episode == train_episodes - 1:
            env.save()
        

    """
    코드 실행 안되면 아랫부분 사용할 것
    """

#    #memory = Memory(training_batch_size)
#
#    for episode in range(train_episodes):
#        state = env.reset(env_steps_size = training_batch_size)
#        #state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#        
#        # Create episode minibatch
#        states, actions, rewards, dones, next_states, log_prob_olds = [], [], [], [], [], []
#        for _ in range(training_batch_size):
#            env.render(visualize)
#            action, log_prob_old = env.act(state, testmode=False) 
#            next_state, reward, done = env.step(action)
#            if done:
#                break
#            env.memory.push(state, action, next_state, reward)
#            # Store (next_state, action, reward) in the episode minibatch
#            states.append(np.expand_dims(state, axis=0))
#            next_states.append(np.expand_dims(next_state, axis=0))
#            #next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
#            actions.append(action)
#            rewards.append(reward)
#            dones.append(done)
#            log_prob_olds.append(log_prob_old)
#            
#            # Update state
#            state = next_state
#        
#        # Timestep 내의 모든 state, action, reward, done, next_state 확보했음
#        # 확보한 애들을 optimise에 던져줘야 함
#        #env.optimize_model(states, actions, rewards, dones, next_states, log_prob_olds)
#        env.optimize_model()
#        print(f"Episode {episode} net_worth: {env.net_worth}")
#        if episode == train_episodes - 1:
#            env.save()
#

def test_agent(env, visualize=False, test_episodes=10):
    env.load() # load the model
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, log_prob_old = env.act(state, testmode=True)
            next_state, reward, done = env.step(action)
            print(f"Episode {episode} net_worth: {env.net_worth}")
            if env.current_step == env.end_step:
                #average_net_worth += env.net_worth
                break
            
    #print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))
