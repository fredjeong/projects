import numpy as np
import random
from collections import deque, namedtuple
import copy
import torch
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter
import gymnasium as gym

from utils_ddpg import TradingGraph, ReplayMemory, init_weights #, OU_noise
from model import Actor, Critic

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TradingEnv(gym.Env):
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, render_range = 100):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.render_range = render_range # render range in visualization

        self.BATCH_SIZE = 10
        self.GAMMA = 0.99
        self.TAU = 0.1
        self.LR = 1e-4
        self.actor_loss = 0
        self.critic_loss = 0
        self.actor_losses = []
        self.critic_losses = []
        self.C = 24 * 7 * 2# 한달마다 거래전략 수정

        #self.OU = OU_noise(1)

    
        self.memory = ReplayMemory(100)

        # Action space from -1 to 1. -1
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,)) 

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

        self.Actor_target = Actor().to(device)
        #self.Actor_target.apply(init_weights)
        self.Actor_behaviour = copy.deepcopy(self.Actor_target).to(device)
        self.Critic_target = Critic().to(device)
        #self.Critic_target.apply(init_weights)
        self.Critic_behaviour = copy.deepcopy(self.Critic_target).to(device)


        self.actor_optimizer = optim.Adam(self.Actor_behaviour.parameters(), lr=self.LR)
        self.critic_optimizer = optim.Adam(self.Critic_behaviour.parameters(), lr=self.LR)

    # Create tensorboard writer
#    def create_writer(self):
#        self.replay_count = 0
#        self.writer = SummaryWriter(comment="Crypto_trader")

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.visualization = TradingGraph(render_range=self.render_range) # init visualization
        self.trades = deque(maxlen=self.render_range) # limited orders memory for visualization
        
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0

        if env_steps_size > 0: # used for training dataset
            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else: # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps
            
        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            self.market_history.append([self.df.loc[current_step, 'open'],
                                        self.df.loc[current_step, 'high'],
                                        self.df.loc[current_step, 'low'],
                                        self.df.loc[current_step, 'close'],
                                        self.df.loc[current_step, 'volume']
                                        ])

        state = np.concatenate((self.market_history, self.orders_history), axis=1)
        return state

    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'open'],
                                    self.df.loc[self.current_step, 'high'],
                                    self.df.loc[self.current_step, 'low'],
                                    self.df.loc[self.current_step, 'close'],
                                    self.df.loc[self.current_step, 'volume']
                                    ])
        obs = np.concatenate((self.market_history, self.orders_history), axis=1)

        return obs

    # Execute one time step within the environment
    def step(self, action):
        #action = action.cpu()
        action = action.item()

        #action = action.cpu() # 새로 추가한 부분
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # Set the current price to a random price between open and close
        current_price = random.uniform(
            self.df.loc[self.current_step, 'open'],
            self.df.loc[self.current_step, 'close'])
        date = self.df.loc[self.current_step, 'date_open'] # for visualization
        high = self.df.loc[self.current_step, 'high'] # for visualization
        low = self.df.loc[self.current_step, 'low'] # for visualization
        
        if action == 0: # Hold
            pass
        
        elif action > 0 and self.balance > 0:#self.initial_balance/100: # 0이 아닌 이유: 가끔 오류가 날 때 있음
            self.crypto_bought = self.balance * action / current_price # 현금 * action으로 정해진 비율만큼 이용해서 구매
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_bought, 'Type': "buy"})
        
        elif action < 0 and self.crypto_held * abs(action) > 0: 
            self.crypto_sold = self.crypto_held * abs(action) 
            self.balance += self.crypto_sold * abs(action) * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_sold, 'Type': "sell"})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Calculate reward
        reward = self.net_worth - self.prev_net_worth

        if self.net_worth <= self.initial_balance / 2:
            done = True
        else:
            done = False

        obs = self._next_observation()
        #obs = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        return obs, reward, done

    # render environment
    def render(self, visualize=False):
        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
        if visualize:
            date = self.df.loc[self.current_step, 'date_open']
            open = self.df.loc[self.current_step, 'open']
            close = self.df.loc[self.current_step, 'close']
            high = self.df.loc[self.current_step, 'high']
            low = self.df.loc[self.current_step, 'low']
            volume = self.df.loc[self.current_step, 'volume']

            # Render the environment to the screen
            self.visualization.render(date, open, high, low, close, volume, self.net_worth, self.trades)
    
    def act(self, state, testmode): # select_action에 대응
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        #with torch.no_grad():
        #    action = self.Actor_behaviour(state) 
        action = self.Actor_behaviour(state) 

        #action = action.cpu()
        #action = action.item()
        action_with_noise = torch.clamp(action + torch.FloatTensor(1).uniform_(-1,1).to(device=device) * 0.2, -1., 1.)

        #if action_with_noise > 1:
        #    action_with_noise = 1
        #elif action_with_noise < -1:
        #    action_with_noise = -1
        
        if testmode == False:
            #return torch.tensor([[action_with_noise]], device=device, dtype=torch.long)
            return action_with_noise
        else:
            #return torch.tensor([[action]], device=device, dtype=torch.long)
            return action


        #if sample > eps_threshold:
        #    with torch.no_grad():
        #        return self.Actor(state).max(1).indices.view(1,1)
        #else:
        #    return torch.tensor([[np.random.choice(self.action_space)]], device=device, dtype=torch.long)
    

    def save(self, name="ddpg"):
        torch.save(self.Actor_target.state_dict(), f'./ddpg/{name}_Actor.h5')
        torch.save(self.Critic_target.state_dict(), f'./ddpg/{name}_Critic.h5')

    def load(self, name="ddpg"):
        self.Actor_target.load_state_dict(torch.load(f'./ddpg/{name}_Actor.h5', weights_only=True))
        self.Critic_target.load_state_dict(torch.load(f'./ddpg/{name}_Critic.h5', weights_only=True))
        
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE * 10:
            return
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        #action_batch = action_batch.squeeze()

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #state_action_values = self.Actor(state_batch).gather(1, action_batch)
        #state_action_values = self.Actor(state_batch).gather(1, action_batch) 
        state_action_values = self.Critic_target(state_batch, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        #with torch.no_grad():
        #    #next_state_values[non_final_mask] = self.Actor(non_final_next_states, state_action_values).max(1).values
        #    next_state_values = self.Critic_behaviour(non_final_next_states, self.Actor_behaviour(non_final_next_states))#.max(1).values
        #    next_state_values = next_state_values.squeeze(1)
        
        next_state_values = next_state_values.unsqueeze(1)
        next_state_values[non_final_mask] = self.Critic_behaviour(non_final_next_states, self.Actor_behaviour(non_final_next_states))#.max(1).values
        next_state_values = next_state_values.squeeze()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)


        self.Critic_behaviour.train()
        critic_criterion = nn.HuberLoss()
        
        self.critic_optimizer.zero_grad()
        self.critic_loss = critic_criterion(state_action_values, expected_state_action_values)
        #torch.nn.utils.clip_grad_value_(self.Critic_behaviour.parameters(), 100)
        self.critic_loss.backward()
        self.critic_optimizer.step()

        self.Actor_behaviour.train()
        self.actor_optimizer.zero_grad()
        self.actor_loss = -torch.mean(self.Critic_behaviour(state_batch, self.Actor_behaviour(state_batch)))
        #torch.nn.utils.clip_grad_value_(self.Actor_behaviour.parameters(), 100)
        self.actor_loss.backward()
        self.actor_optimizer.step()

        if self.current_step % self.C == 0:
            self.soft_update_target(self.Critic_target, self.Critic_behaviour)
            self.soft_update_target(self.Actor_target, self.Actor_behaviour)

    def soft_update_target(self, target, original):
        for target_parameters, original_parameters in zip(target.parameters(), original.parameters()):
            target_parameters.data.copy_((1 - self.TAU) * target_parameters.data + self.TAU * original_parameters.data)



def train_agent(env, visualize=False, train_episodes=1, training_batch_size=500):
    #env.create_writer() # create TensorBoard writer
    #total_average = deque(maxlen=100) # save recent 100 episodes net worth
    #best_average = 0 # used to track best average net worth
    
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in range(training_batch_size):
            env.render(visualize)
            action = env.act(state, testmode=False)
            next_state, reward, done = env.step(action)

            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
            action = action.item()
            #if action == int(action):
            #    pass
            #else:
            #    action = action.item()

            action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            if done:
                break
            #action_onehot = np.zeros(3)
            #action_onehot[action] = 1
            env.memory.push(state, action, next_state, reward) # Store the transition in memory
            #next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            state = next_state
            env.optimize_model() # perform one step of the optimization on the policy network

            print(f"Action: {action.item()}")
            print(f"Episode {episode} net_worth: {env.net_worth} step: {env.current_step}")
            print(f"Episode {episode} Actor loss: {env.actor_loss} Critic loss: {env.critic_loss}")
        if episode == train_episodes - 1:
            env.save()

def test_agent(env, visualize=False, test_episodes=1):
    env.load() # load the model
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = env.act(state, testmode=True)
            state, reward, done = env.step(action)
            print(f"Episode {episode} net_worth: {env.net_worth}")
            if env.current_step == env.end_step:
                break

#import numpy as np
#import random
#from collections import deque, namedtuple
#import copy
#import torch
#import torch.nn as nn
#import torch.optim as optim
##from tensorboardX import SummaryWriter
#import gymnasium as gym
#
#from utils_ddpg import TradingGraph, ReplayMemory#, OU_noise
#from model import Actor, Critic
#
#device = torch.device(
#    "cuda" if torch.cuda.is_available() else
#    "mps" if torch.backends.mps.is_available() else
#    "cpu"
#)
#
#class TradingEnv:
#    # A custom Bitcoin trading environment
#    def __init__(self, df, initial_balance=1000, lookback_window_size=50, render_range = 100):
#        # Define action space and state size and other custom parameters
#        self.df = df.dropna().reset_index()
#        self.df_total_steps = len(self.df)-1
#        self.initial_balance = initial_balance
#        self.lookback_window_size = lookback_window_size
#        self.render_range = render_range # render range in visualization
#
#        self.BATCH_SIZE = 64
#        self.GAMMA = 0.99
#        self.TAU = 0.1
#        self.LR = 1e-4
#
#        #self.OU = OU_noise(1)
#
#    
#        self.memory = ReplayMemory(1000)
#
#        # Action space from -1 to 1. -1
#        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,)) 
#
#        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
#        self.orders_history = deque(maxlen=self.lookback_window_size)
#        
#        # Market history contains the OHCL values for the last lookback_window_size prices
#        self.market_history = deque(maxlen=self.lookback_window_size)
#
#        # State size contains Market+Orders history for the last lookback_window_size steps
#        self.state_size = (self.lookback_window_size, 10)
#
#        self.Actor_target = Actor().to(device)
#        self.Actor_behaviour = copy.deepcopy(self.Actor_target).to(device)
#        self.Critic_target = Critic().to(device)
#        self.Critic_behaviour = copy.deepcopy(self.Critic_target).to(device)
#
#        self.actor_optimizer = optim.Adam(self.Actor_behaviour.parameters(), lr=self.LR)
#        self.critic_optimizer = optim.Adam(self.Critic_behaviour.parameters(), lr=self.LR)
#
#    # Create tensorboard writer
##    def create_writer(self):
##        self.replay_count = 0
##        self.writer = SummaryWriter(comment="Crypto_trader")
#
#    # Reset the state of the environment to an initial state
#    def reset(self, env_steps_size = 0):
#        self.visualization = TradingGraph(render_range=self.render_range) # init visualization
#        self.trades = deque(maxlen=self.render_range) # limited orders memory for visualization
#        
#        self.balance = self.initial_balance
#        self.net_worth = self.initial_balance
#        self.prev_net_worth = self.initial_balance
#        self.crypto_held = 0
#        self.crypto_sold = 0
#        self.crypto_bought = 0
#
#        if env_steps_size > 0: # used for training dataset
#            self.start_step = random.randint(self.lookback_window_size, self.df_total_steps - env_steps_size)
#            self.end_step = self.start_step + env_steps_size
#        else: # used for testing dataset
#            self.start_step = self.lookback_window_size
#            self.end_step = self.df_total_steps
#            
#        self.current_step = self.start_step
#
#        for i in reversed(range(self.lookback_window_size)):
#            current_step = self.current_step - i
#            self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
#            self.market_history.append([self.df.loc[current_step, 'open'],
#                                        self.df.loc[current_step, 'high'],
#                                        self.df.loc[current_step, 'low'],
#                                        self.df.loc[current_step, 'close'],
#                                        self.df.loc[current_step, 'volume']
#                                        ])
#
#        state = np.concatenate((self.market_history, self.orders_history), axis=1)
#        return state
#
#    # Get the data points for the given current_step
#    def _next_observation(self):
#        self.market_history.append([self.df.loc[self.current_step, 'open'],
#                                    self.df.loc[self.current_step, 'high'],
#                                    self.df.loc[self.current_step, 'low'],
#                                    self.df.loc[self.current_step, 'close'],
#                                    self.df.loc[self.current_step, 'volume']
#                                    ])
#        obs = np.concatenate((self.market_history, self.orders_history), axis=1)
#
#        return obs
#
#    # Execute one time step within the environment
#    def step(self, action):
#        if action == int(action):
#            pass
#        else:
#            action = action.item()
#
#        #action = action.cpu() # 새로 추가한 부분
#        self.crypto_bought = 0
#        self.crypto_sold = 0
#        self.current_step += 1
#
#        # Set the current price to a random price between open and close
#        current_price = random.uniform(
#            self.df.loc[self.current_step, 'open'],
#            self.df.loc[self.current_step, 'close'])
#        date = self.df.loc[self.current_step, 'date_open'] # for visualization
#        high = self.df.loc[self.current_step, 'high'] # for visualization
#        low = self.df.loc[self.current_step, 'low'] # for visualization
#        
#        if action == 0: # Hold
#            pass
#        
#        elif action > 0 and self.balance > self.initial_balance/100: # 0이 아닌 이유: 가끔 오류가 날 때 있음
#            self.crypto_bought = self.balance * action / current_price # 현금 * action으로 정해진 비율만큼 이용해서 구매
#            self.balance -= self.crypto_bought * current_price
#            self.crypto_held += self.crypto_bought
#            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_bought, 'Type': "buy"})
#        
#        elif action < 0 and self.crypto_held * abs(action) > 0: 
#            self.crypto_sold = self.crypto_held * abs(action) 
#            self.balance += self.crypto_sold * abs(action) * current_price
#            self.crypto_held -= self.crypto_sold
#            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_sold, 'Type': "sell"})
#
#        self.prev_net_worth = self.net_worth
#        self.net_worth = self.balance + self.crypto_held * current_price
#
#        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
#
#        # Calculate reward
#        reward = self.net_worth - self.prev_net_worth
#
#        if self.net_worth <= self.initial_balance/2:
#            done = True
#        else:
#            done = False
#
#        obs = self._next_observation()
#        #obs = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#        
#        return obs, reward, done
#
#    # render environment
#    def render(self, visualize=False):
#        #print(f'Step: {self.current_step}, Net Worth: {self.net_worth}')
#        if visualize:
#            date = self.df.loc[self.current_step, 'date_open']
#            open = self.df.loc[self.current_step, 'open']
#            close = self.df.loc[self.current_step, 'close']
#            high = self.df.loc[self.current_step, 'high']
#            low = self.df.loc[self.current_step, 'low']
#            volume = self.df.loc[self.current_step, 'volume']
#
#            # Render the environment to the screen
#            self.visualization.render(date, open, high, low, close, volume, self.net_worth, self.trades)
#    
#    def act(self, state, testmode): # select_action에 대응
#        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#        with torch.no_grad():
#            action = self.Actor_behaviour(state) 
#
#        action = action.cpu()
#        #action = action.item()
#        action_with_noise = action + np.random.rand(1) * 0.1            
#
#        if action_with_noise > 1:
#            action_with_noise = 1
#        elif action_with_noise < -1:
#            action_with_noise = -1
#        
#        if testmode == False:
#            #return torch.tensor([[action_with_noise]], device=device, dtype=torch.long)
#            return action_with_noise
#        else:
#            #return torch.tensor([[action]], device=device, dtype=torch.long)
#            return action
#
#
#        #if sample > eps_threshold:
#        #    with torch.no_grad():
#        #        return self.Actor(state).max(1).indices.view(1,1)
#        #else:
#        #    return torch.tensor([[np.random.choice(self.action_space)]], device=device, dtype=torch.long)
#    
#
#    def save(self, name="ddpg"):
#        torch.save(self.Actor_target.state_dict(), f'./ddpg/{name}_Actor.h5')
#        torch.save(self.Critic_target.state_dict(), f'./ddpg/{name}_Critic.h5')
#
#    def load(self, name="ddpg"):
#        self.Actor_target.load_state_dict(torch.load(f'./ddpg/{name}_Actor.h5', weights_only=True))
#        self.Critic_target.load_state_dict(torch.load(f'./ddpg/{name}_Critic.h5', weights_only=True))
#        
#    def optimize_model(self):
#        if len(self.memory) < self.BATCH_SIZE:
#            return
#        Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))
#        transitions = self.memory.sample(self.BATCH_SIZE)
#        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
#        # detailed explanation). This converts batch-array of Transitions
#        # to Transition of batch-arrays.
#        batch = Transition(*zip(*transitions))
#
#        # Compute a mask of non-final states and concatenate the batch elements
#        # (a final state would've been the one after which simulation ended)
#        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
#                                            batch.next_state)), device=device, dtype=torch.bool)
#        non_final_next_states = torch.cat([s for s in batch.next_state
#                                                    if s is not None])
#        state_batch = torch.cat(batch.state)
#        action_batch = torch.cat(batch.action)
#        reward_batch = torch.cat(batch.reward)
#
#        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
#        # columns of actions taken. These are the actions which would've been taken
#        # for each batch state according to policy_net
#        #state_action_values = self.Actor(state_batch).gather(1, action_batch)
#        #state_action_values = self.Actor(state_batch).gather(1, action_batch) 
#        state_action_values = self.Critic_behaviour(state_batch, action_batch).gather(1, action_batch) 
#
#        # Compute V(s_{t+1}) for all next states.
#        # Expected values of actions for non_final_next_states are computed based
#        # on the "older" target_net; selecting their best reward with max(1).values
#        # This is merged based on the mask, such that we'll have either the expected
#        # state value or 0 in case the state was final.
#        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
#        with torch.no_grad():
#            #next_state_values[non_final_mask] = self.Actor(non_final_next_states, state_action_values).max(1).values
#            next_state_values[non_final_mask] = self.Critic_target(non_final_next_states, self.Actor_target(non_final_next_states))#.max(1).values
#        # Compute the expected Q values
#        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch
#
#
#        # Compute Huber loss
#        critic_criterion = nn.MSELoss()
#        critic_loss = critic_criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#
#        # Optimize the model
#        self.critic_optimizer.zero_grad()
#        critic_loss.backward()
#
#        # In-place gradient clipping
#        #torch.nn.utils.clip_grad_value_(self.Actor.parameters(), 100)
#        self.critic_optimizer.step()
#
#        
#        actor_loss = -torch.mean(self.Critic_behaviour(state_batch, self.Actor_behaviour(state_batch)))
#        self.actor_optimizer.zero_grad()
#        actor_loss.backward()
#        self.actor_optimizer.step()
#
#        self.soft_update_target(self.Critic_target, self.Critic_behaviour)
#        self.soft_update_target(self.Actor_target, self.Actor_behaviour)
#
#    def soft_update_target(self, target, original):
#        for target_parameters, original_parameters in zip(target.parameters(), original.parameters()):
#            target_parameters.data.copy_((1 - self.TAU) * target_parameters.data + self.TAU * original_parameters.data)
#        
#
#def train_agent(env, visualize=False, train_episodes=1, training_batch_size=500):
#    #env.create_writer() # create TensorBoard writer
#    #total_average = deque(maxlen=100) # save recent 100 episodes net worth
#    #best_average = 0 # used to track best average net worth
#    memory = ReplayMemory(1000)
#    
#    for episode in range(train_episodes):
#        state = env.reset(env_steps_size = training_batch_size)
#        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
#        
#        for t in range(training_batch_size):
#            env.render(visualize)
#            action = env.act(state, testmode=False)
#            next_state, reward, done = env.step(action)
#            if done:
#                break
#            #action_onehot = np.zeros(3)
#            #action_onehot[action] = 1
#            memory.push(state, action, next_state, reward) # Store the transition in memory
#            #next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
#            state = next_state
#            env.optimize_model() # perform one step of the optimization on the policy network
#
#            print(f"net_worth: {env.net_worth}, step: {env.current_step}")
#        if episode == train_episodes - 1:
#            env.save()
#    
#def test_agent(env, visualize=True, test_episodes=1):
#    env.load() # load the model
#    for episode in range(test_episodes):
#        state = env.reset()
#        while True:
#            env.render(visualize)
#            action = env.act(state, testmode=True)
#            state, reward, done = env.step(action)
#            print(f"Episode {episode} net_worth: {env.net_worth}")
#            if env.current_step == env.end_step:
#                break
#
#