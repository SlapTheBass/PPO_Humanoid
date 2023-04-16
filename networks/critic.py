import torch
import torch.nn as nn
import sys

from pathlib import Path

file_dir = Path(__file__).parent
module_path = str(file_dir.joinpath('..'))
sys.path.append(module_path)

from utilities.config import *


#*************************** CRITIC ***********************************

class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, HIDDEN_LAYERS_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_LAYERS_SIZE, 1),
        )

    def forward(self, x):
        return self.value(x)
    

if __name__ == "__main__":

    from environment.env import *

    env = MakeMainEnvironment()
    device = IsCudaEnabled()

    net_crt = ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_crt)