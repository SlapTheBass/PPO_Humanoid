import torch
import torch.nn as nn

from utilities.config import *


#*************************** ACTOR ***********************************

class ModelActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(ModelActor, self).__init__()

        self.mu = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_LAYERS_SIZE[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS_SIZE[0], HIDDEN_LAYERS_SIZE[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS_SIZE[1], act_size),
            nn.ReLU()
        )
        self.logstd = nn.Parameter(torch.zeros(act_size))

    def forward(self, x):
        return self.mu(x)
    


if __name__ == "__main__":

    from environment.env import *

    env = GetMainEnvironment()
    device = IsCudaEnabled()

    net_act = ModelActor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
    print(net_act)