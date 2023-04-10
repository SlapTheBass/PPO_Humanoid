import torch
import torch.nn as nn

from config import *


#*************************** CRITIC ***********************************

class ModelCritic(nn.Module):
    def __init__(self, obs_size):
        super(ModelCritic, self).__init__()

        self.value = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_LAYERS_SIZE[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS_SIZE[0], HIDDEN_LAYERS_SIZE[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS_SIZE[1], 1),
        )

    def forward(self, x):
        return self.value(x)
    

if __name__ == "__main__":

    from env import *

    env = GetMainEnvironment()
    device = IsCudaEnabled()

    net_crt = ModelCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net_crt)