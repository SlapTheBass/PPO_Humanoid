import torch
import torch.nn as nn
import sys

from pathlib import Path

file_dir = Path(__file__).parent
module_path = str(file_dir.joinpath('..'))
sys.path.append(module_path)

from utilities.config import *


#*************************** ACTOR ***********************************

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, act_size),
            nn.Tanh(),
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)
    


if __name__ == "__main__":

    from environment.env import *

    env = MakeMainEnvironment()
    device = IsCudaEnabled()

    net_act = ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net_act)