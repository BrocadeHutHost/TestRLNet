import torch
import torch.nn as nn
import torch.optim as optim

class Network(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(n_states, 8)  
        self.fc2 = nn.Linear(8, n_actions) 
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)              
        return x