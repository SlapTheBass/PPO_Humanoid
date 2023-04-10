import ptan
import numpy as np
import math
import torch
import torch.nn as nn

from config import *

HIDDEN_LAYERS_SIZE = {512, 256}


#*************************** MODELS ***********************************

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
    

#**************************** FUNCTIONS ********************************

def test_net(net, env, device, count=10):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def calc_logprob(mu_v, logstd_v, actions_v):
    p1 = - ((mu_v - actions_v) ** 2) / (2*torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))
    return p1 + p2



def calc_adv_ref(trajectory, net_crt, states_v, device):
    
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()

    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + GAMMA * next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv)))
    ref_v = torch.FloatTensor(list(reversed(result_ref)))
    return adv_v.to(device), ref_v.to(device)