import numpy as np
import random
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
#from tensorboardX import SummaryWriter
from utils_dqn import TradingGraph, ReplayMemory, init_weights
from model import DQN
import copy

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class TradingEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, render_range = 100):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df)-1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.render_range = render_range # render range in visualization

        self.batch_size = 64
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.1
        self.lr = 1e-4
        self.C = 24*30 # Update target every C steps (Update every week 원래 한달이었고 이거 성능 잘나왔음)
        self.losses = []
        self.ratio = 0.1

        self.memory = ReplayMemory(1000)
        self.loss = 0

        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])
        n_actions = len(self.action_space)

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)
        
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        # State size contains Market+Orders history for the last lookback_window_size steps
        self.state_size = (self.lookback_window_size, 10)

        self.Target = DQN(n_actions, self.state_size).to(device)
        self.Target.apply(init_weights)
        self.Behaviour = copy.deepcopy(self.Target).to(device)
        

        self.optimizer = optim.Adam(self.Behaviour.parameters(), lr=self.lr)


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

        elif action == 1 and self.balance > self.initial_balance/100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance * self.ratio / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_bought, 'Type': "buy"})

        elif action == 2 and self.crypto_held>0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held * self.ratio
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_sold, 'Type': "sell"})

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
        #Write_to_file(Date, self.orders_history[-1])

        # Calculate reward
        reward = (self.net_worth - self.prev_net_worth)

        if self.net_worth <= self.initial_balance / 2:
            #reward = -10000
            done = True
        else:
            done = False

        obs = self._next_observation()
        
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
        sample = random.random()
        # 초반에는 엡실론 값을 높게 하여 최대한 다양한 경험을 해보도록 하고, 점점 그 값을 낮춰가며 신경망이 결정하는 비율을 높인다
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.current_step / self.eps_decay) 
        
        if testmode == False:
            if sample > eps_threshold:
                with torch.no_grad():
                    return self.Behaviour(state).max(1).indices.view(1,1)
            else:
                return torch.tensor([[np.random.choice(self.action_space)]], device=device, dtype=torch.long)
        else:
            return self.Target(state).max(1).indices.view(1,1)

    def save(self, name="dqn"):
        torch.save(self.Target.state_dict(), f'./dqn/{name}_Target.h5')
        torch.save(self.Behaviour.state_dict(), f'./dqn/{name}_Behaviour.h5')

    def load(self, name="dqn"):
        self.Behaviour.load_state_dict(torch.load(f'./dqn/{name}_Behaviour.h5', weights_only=True))
        #self.Behaviour.load_state_dict(torch.load(f'./dqn/{name}_Behaviour.h5', weights_only=True))
        self.Target.load_state_dict(torch.load(f'./dqn/{name}_Target.h5', weights_only=True))
        
    def optimize_model(self):
        # Experience replay 메모리의 길이가 batch size (10)의 5배가 되기 전까지는 샘플링을 하지 않는다.
        if len(self.memory) < self.batch_size * 5:
            return
        Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        #with torch.no_grad():
        #    state_action_values = self.Target(state_batch).gather(1, action_batch)
        state_action_values = self.Target(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        #with torch.no_grad():
        #    next_state_values[non_final_mask] = self.Behaviour(non_final_next_states).max(1).values
        next_state_values[non_final_mask] = self.Behaviour(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)

        # Compute Huber loss
        self.Behaviour.train()
        criterion = nn.HuberLoss()
        #criterion = nn.HuberLoss()
        self.loss = criterion(state_action_values, expected_state_action_values)#.unsqueeze(1)
        self.losses.append(self.loss)
        # Optimize the model
        self.optimizer.zero_grad()
        torch.nn.utils.clip_grad_value_(self.Behaviour.parameters(), 100)
        self.loss.backward()
        # In-place gradient clipping
        
        self.optimizer.step()
        
        # Update the target network every C step
        if self.current_step % self.C == 0:
            self.soft_update_target(self.Target, self.Behaviour)
    
    def soft_update_target(self, target, behaviour):
        for target_parameter, behaviour_parameter in zip(target.parameters(), behaviour.parameters()):
            target_parameter.data.copy_((1 - self.tau) * target_parameter.data + self.tau * behaviour_parameter)

def train_agent(env, visualize=False, train_episodes=1, training_batch_size=500):
    
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        for t in range(training_batch_size):
            env.render(visualize)
            action = env.act(state, testmode=False)
            next_state, reward, done = env.step(action)
            # 추가한 부분
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            reward = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)
            if done:
                break
            #action_onehot = np.zeros(3)
            #action_onehot[action] = 1
            env.memory.push(state, action, next_state, reward) # Store the transition in memory
            state = next_state
            env.optimize_model() # perform one step of the optimization on the policy network

        #torch.cat(env.losses, dim=0)
        print(f"Episode: {episode} Net worth: {env.net_worth}")
        print(f"Episode {episode} loss: {env.loss}")
        #print(f"Episode {episode} average loss: {torch.mean(env.losses)}")
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
            print(f"Episode {episode} net_worth:, {env.net_worth}")
            #print(f"Episode {episode} average loss: {np.mean(env.losses)}")
            if env.current_step == env.end_step:
                break

#class TradingEnv:
#    # A custom Bitcoin trading environment
#    def __init__(self, df, initial_balance=10000, lookback_window_size=50, render_range = 100):
#        # Define action space and state size and other custom parameters
#        self.df = df.dropna().reset_index()
#        self.df_total_steps = len(self.df)-1
#        self.initial_balance = initial_balance
#        self.lookback_window_size = lookback_window_size
#        self.render_range = render_range # render range in visualization
#
#        self.batch_size = 64
#        self.gamma = 0.99
#        self.eps_start = 0.9
#        self.eps_end = 0.05
#        self.eps_decay = 1000
#        self.tau = 0.1
#        self.lr = 1e-4
#        self.C = 10 # Update target every C steps
#        self.losses = []
#        self.ratio = 0.1
#
#        self.memory = ReplayMemory(1000)
#
#        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
#        self.action_space = np.array([0, 1, 2])
#        n_actions = len(self.action_space)
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
#        self.Target = DQN(n_actions, self.state_size).to(device)
#        self.Behaviour = DQN(n_actions, self.state_size).to(device)
#
#        self.optimizer = optim.Adam(self.Behaviour.parameters(), lr=self.lr)
#
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
#        return obs
#
#    # Execute one time step within the environment
#    def step(self, action):
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
#        elif action == 1 and self.balance > self.initial_balance/100: # Buy
#            # Buy with 100% of current balance
#            self.crypto_bought = self.balance * self.ratio / current_price
#            self.balance -= self.crypto_bought * current_price
#            self.crypto_held += self.crypto_bought
#            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_bought, 'Type': "buy"})
#
#        elif action == 2 and self.crypto_held > 0: # 
#            # Sell 100% of current crypto held
#            self.crypto_sold = self.crypto_held * self.ratio
#            self.balance += self.crypto_sold * current_price
#            self.crypto_held -= self.crypto_sold
#            self.trades.append({'Date' : date, 'High' : high, 'Low' : low, 'Total': self.crypto_sold, 'Type': "sell"})
#
#        self.prev_net_worth = self.net_worth
#        self.net_worth = self.balance + self.crypto_held * current_price
#
#        self.orders_history.append([self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
#        #Write_to_file(Date, self.orders_history[-1])
#
#        # Calculate reward
#        reward = (self.net_worth - self.prev_net_worth)
#
#        if self.net_worth <= self.initial_balance / 2:
#            #reward = -10000
#            done = True
#        else:
#            done = False
#
#        obs = self._next_observation()
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
#        sample = random.random()
#        # 초반에는 엡실론 값을 높게 하여 최대한 다양한 경험을 해보도록 하고, 점점 그 값을 낮춰가며 신경망이 결정하는 비율을 높인다
#        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * np.exp(-1 * self.current_step / self.eps_decay) 
#        
#        if testmode == False:
#            if sample > eps_threshold:
#                with torch.no_grad():
#                    return self.Behaviour(state).max(1).indices.view(1,1)
#            else:
#                return torch.tensor([[np.random.choice(self.action_space)]], device=device, dtype=torch.long)
#        else:
#            return self.Target(state).max(1).indices.view(1,1)
#
#    def save(self, name="dqn"):
#        torch.save(self.Target.state_dict(), f'./dqn/{name}_Target.h5')
#        torch.save(self.Behaviour.state_dict(), f'./dqn/{name}_Behaviour.h5')
#
#    def load(self, name="dqn"):
#        self.Target.load_state_dict(torch.load(f'./dqn/{name}_Target.h5', weights_only=True))
#        #self.Behaviour.load_state_dict(torch.load(f'./dqn/{name}_Behaviour.h5', weights_only=True))
#        self.Behaviour.load_state_dict(torch.load(f'./dqn/{name}_Target.h5', weights_only=True))
#        
#    def optimize_model(self):
#        if len(self.memory) < self.batch_size:
#            return
#        Transition = namedtuple('Transition',
#                        ('state', 'action', 'next_state', 'reward'))
#        transitions = self.memory.sample(self.batch_size)
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
#        state_action_values = self.Behaviour(state_batch).gather(1, action_batch)
#
#        # Compute V(s_{t+1}) for all next states.
#        # Expected values of actions for non_final_next_states are computed based
#        # on the "older" target_net; selecting their best reward with max(1).values
#        # This is merged based on the mask, such that we'll have either the expected
#        # state value or 0 in case the state was final.
#        next_state_values = torch.zeros(self.batch_size, device=device)
#        with torch.no_grad():
#            next_state_values[non_final_mask] = self.Target(non_final_next_states).max(1).values
#        # Compute the expected Q values
#        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
#
#        # Compute Huber loss
#        criterion = nn.MSELoss()
#        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
#        self.losses.append(loss)
#        # Optimize the model
#        self.optimizer.zero_grad()
#        loss.backward()
#        # In-place gradient clipping
#        #torch.nn.utils.clip_grad_value_(self.Policy.parameters(), 100)
#        self.optimizer.step()
#        
#        # Update the target network every C step
#        if self.current_step % self.C == 0:
#            self.soft_update_target(self.Target, self.Behaviour)
#    
#    def soft_update_target(self, target, behaviour):
#        for target_parameter, behaviour_parameter in zip(target.parameters(), behaviour.parameters()):
#            target_parameter.data.copy_((1 - self.tau) * target_parameter.data + self.tau * behaviour_parameter)
#
#def train_agent(env, visualize=False, train_episodes=1, training_batch_size=500):
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
#            state = next_state
#            env.optimize_model() # perform one step of the optimization on the policy network
#
#        print(f"Episode: {episode} Net worth: {env.net_worth}")
#        print(f"Episode {episode} average loss: {np.mean(env.losses)}")
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
#            print(f"Episode {episode} net_worth:, {env.net_worth}")
#            #print(f"Episode {episode} average loss: {np.mean(env.losses)}")
#            if env.current_step == env.end_step:
#                break
#
#