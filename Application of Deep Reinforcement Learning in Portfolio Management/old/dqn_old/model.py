import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class DQN(nn.Module):
    def __init__(self, action_space, state_size):
        super(DQN, self).__init__()
        self.action_space = action_space
        self.state_size = state_size
        self.state_size = self.state_size[0] * self.state_size[1]

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(500, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 64)
        self.layer4 = nn.Linear(64, self.action_space)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return F.sigmoid(self.layer4(x))