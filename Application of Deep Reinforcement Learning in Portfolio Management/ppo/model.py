import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class Actor(nn.Module): # action return
    def __init__(self):
        super(Actor, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(500, 512) # state_dim = 500
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.mu_layer = nn.Linear(64, 1) # action_dim = 1
        self.log_std_layer = nn.Linear(64, 1) # action_dim = 1
        
        #self.layer1 = nn.Linear(500, 64) # state_dim = 500
        #self.layer2 = nn.Linear(64, 64)
        #self.layer3 = nn.Linear(64, 64)
        #self.mu_layer = nn.Linear(64, 1) # action_dim = 1
        #self.log_std_layer = nn.Linear(64, 1) # action_dim = 1

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))

        mu = self.mu_layer(x)
        log_std = torch.tanh(self.log_std_layer(x)) # -1에서 1 사이로 표준편차의 로그값 반환

        return mu, log_std.exp()

class Critic(nn.Module): # value return
    def __init__(self):
        super(Critic, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(500, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)

        return x