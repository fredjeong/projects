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
    def __init__(self):
        super(Actor, self).__init__()
        self.n_actions = 1

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(500, 256)
        self.layer2 = nn.Linear(256, 64)
        self.layer3 = nn.Linear(64, self.n_actions) # continuos action space니까 1이 되어야 하지 않을까? # 
        # action size: 여기서는 비트코인 한 종류만 다루고 있으므로 마지막을 1로 처리
    
    def forward(self, state):
        x = state
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x)) # 뒤에 tanh가 오는데 ReLU를 또 써?
        #x = self.layer3(x)

        return torch.tanh(x) # tanh 함수는 결과값을 -1에서 1 사이로 가두어 준다. 그러므로 continuous action space에 알맞다.

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.n_actions = 1
        # Layer 1
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(500, 256)
        self.batch_norm_1 = nn.BatchNorm1d(256)

        # Layer 2
        # In the second layer the actions will be inserted also 
        self.layer2 = nn.Linear(256 + self.n_actions, 64)
        self.batch_norm_2 = nn.BatchNorm1d(64)

        # Output layer (single value)
        self.layer3 = nn.Linear(64, 1)


    def forward(self, state, action):
        x = state

        # Layer 1
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        # Layer 2
        x = torch.cat((x, action), 1)  # Insert the actions
        x = self.layer2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        return self.layer3(x) # Value라서 x 대신 V라고도 부른다.