import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, action_space, state_size):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.state_size = state_size
        self.state_size = self.state_size[0] * self.state_size[1]

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(500, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, self.action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x)