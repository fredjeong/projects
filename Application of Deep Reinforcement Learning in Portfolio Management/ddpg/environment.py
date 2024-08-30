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
        self.TAU = 0.2
        self.LR = 1e-4
        self.actor_losses = []
        self.critic_losses = []
        self.portfolio_values = []
        self.benchmark = []


        self.memory = ReplayMemory(len(self.df))


        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,)) 
        self.orders_history = deque(maxlen=self.lookback_window_size)
        self.market_history = deque(maxlen=self.lookback_window_size)
        self.state_size = (self.lookback_window_size, 10)

        self.Actor_behaviour = Actor(self.lookback_window_size).to(device)
        self.Actor_target = copy.deepcopy(self.Actor_behaviour).to(device)
        self.Critic_behaviour = Critic(self.lookback_window_size).to(device)
        self.Critic_target = copy.deepcopy(self.Critic_behaviour).to(device)


        self.actor_optimizer = optim.Adam(self.Actor_behaviour.parameters(), lr=self.LR)
        self.critic_optimizer = optim.Adam(self.Critic_behaviour.parameters(), lr=self.LR)

    # Create tensorboard writer
#    def create_writer(self):
#        self.replay_count = 0
#        self.writer = SummaryWriter(comment="Crypto_trader")

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size = 0):
        self.portfolio_values = []
        self.benchmark = []
        
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
        action = self.Actor_behaviour(state) 
        action_with_noise = torch.clamp(action + torch.FloatTensor(1).uniform_(-1,1).to(device=device) * 0.2, -1., 1.)
        
        if testmode == False:
            return action_with_noise
        else:
            return action    

    def save(self, name="ddpg"):
        torch.save(self.Actor_behaviour.state_dict(), f'./ddpg/{name}_Actor.h5')
        torch.save(self.Critic_behaviour.state_dict(), f'./ddpg/{name}_Critic.h5')

    def load(self, name="ddpg"):
        self.Actor_behaviour.load_state_dict(torch.load(f'./ddpg/{name}_Actor.h5', weights_only=False))
        self.Critic_behaviour.load_state_dict(torch.load(f'./ddpg/{name}_Critic.h5', weights_only=False))
        
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
        state_action_values = self.Critic_behaviour(state_batch, action_batch)
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        
        next_state_values = next_state_values.unsqueeze(1)
        next_state_values[non_final_mask] = self.Critic_target(non_final_next_states, self.Actor_target(non_final_next_states))#.max(1).values
        next_state_values = next_state_values.squeeze()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values) + reward_batch
        expected_state_action_values = expected_state_action_values.unsqueeze(1)


        self.Critic_behaviour.train()
        critic_criterion = nn.HuberLoss()
        
        self.critic_optimizer.zero_grad()
        critic_loss = critic_criterion(state_action_values, expected_state_action_values)
        self.critic_losses.append(critic_loss.item())
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.Critic_behaviour.parameters(), 10)
        self.critic_optimizer.step()

        self.Actor_behaviour.train()
        self.actor_optimizer.zero_grad()
        actor_loss = -torch.mean(self.Critic_behaviour(state_batch, self.Actor_behaviour(state_batch)))
        self.actor_losses.append(actor_loss.item())
        actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.Actor_behaviour.parameters(), 10)
        self.actor_optimizer.step()

        self.soft_update_target(self.Critic_target, self.Critic_behaviour)
        self.soft_update_target(self.Actor_target, self.Actor_behaviour)

    def soft_update_target(self, target, original):
        for target_parameters, original_parameters in zip(target.parameters(), original.parameters()):
            target_parameters.data.copy_((1 - self.TAU) * target_parameters.data + self.TAU * original_parameters.data)



def train_agent(env, visualize=False, train_episodes=1, training_batch_size=500):
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
            action = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            
            if done:
                break
            env.memory.push(state, action, next_state, reward) # Store the transition in memory
            state = next_state
            env.portfolio_values.append(env.net_worth)
            env.benchmark.append(env.df.loc[env.current_step, 'close'] * env.initial_balance / env.df.loc[env.start_step, 'close'])

            env.optimize_model() 

            print(f"Action: {action.item()}")
            print(f"Episode {episode} net_worth: {env.net_worth} step: {env.current_step}")
            #print(f"Episode {episode} Actor loss: {env.actor_loss} Critic loss: {env.critic_loss}")
        
        np.savetxt(f'./ddpg/records/benchmark_{episode}.csv', env.benchmark)
        np.savetxt(f'./ddpg/records/portfolio_values_{episode}.csv', env.portfolio_values)
        if episode == train_episodes - 1:
            env.save()
    np.savetxt('./ddpg/records/actor_losses.csv', env.actor_losses)
    np.savetxt('./ddpg/records/critic_losses.csv', env.critic_losses)


def test_agent(env, visualize=False, test_episodes=1):
    env.load() # load the model
    env.Actor_behaviour.eval()
    env.Critic_behaviour.eval()
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action = env.act(state, testmode=True)
            state, reward, done = env.step(action)
            env.portfolio_values.append(env.net_worth)
            print(f"Episode {episode} net_worth: {env.net_worth}")
            if env.current_step == env.end_step:
                break
    np.savetxt('./ddpg/portfolio_values.csv', env.portfolio_values)