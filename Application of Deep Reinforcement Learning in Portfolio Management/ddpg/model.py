import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Actor(nn.Module):
    def __init__(self, lookback_window_size):
        super(Actor, self).__init__()
        self.n_actions = 1
        self.lookback_window_size = lookback_window_size

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(lookback_window_size * 10, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, self.n_actions, bias=False) # continuos action space니까 1이 되어야 하지 않을까? # 
        # action size: 여기서는 비트코인 한 종류만 다루고 있으므로 마지막을 1로 처리
    
    def forward(self, state):
        x = state
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) # 뒤에 tanh가 오는데 ReLU를 또 써?
        #x = self.layer3(x)

        return torch.tanh(self.layer4(x)) # tanh 함수는 결과값을 -1에서 1 사이로 가두어 준다. 그러므로 continuous action space에 알맞다.

class Critic(nn.Module):
    def __init__(self, lookback_window_size):
        super(Critic, self).__init__()
        self.n_actions = 1
    
        self.lookback_window_size = lookback_window_size

        # Layer 1
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(lookback_window_size * 10, 64)
        self.batch_norm_1 = nn.BatchNorm1d(64)

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.layer2 = nn.Linear(64, 64)
        self.batch_norm_2 = nn.BatchNorm1d(64)

        self.layer3 = nn.Linear(64 + self.n_actions, 64)
        self.batch_norm_3 = nn.BatchNorm1d(64)

        # Output layer (single value)
        self.layer4 = nn.Linear(64, 1)


    def forward(self, state, action):
        x = state

        # Layer 1
        x = self.flatten(x)
        x = self.layer1(x)
        #x = self.batch_norm_1(x)
        x = F.relu(x)

        # Layer 2
        x = F.relu(self.layer2(x))
        #x = self.batch_norm_2(x)

        x = torch.cat((x, action), 1)  # Insert the actions
        x = self.layer3(x)
        #x = self.batch_norm_3(x)
        x = F.relu(x)

        return self.layer4(x) # Value라서 x 대신 V라고도 부른다.