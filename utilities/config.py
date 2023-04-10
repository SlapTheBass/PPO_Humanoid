import os
import torch

#*********** SAVE/LOAD paths *****************

SAVE_PATH = " "
LOAD_PATH = " "



#********** HYPER PARAMETERS *****************

GAMMA = 0.97
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 128

TEST_ITERS = 1000


#********* REPLAY PARAMETERS ****************

STEPS = 100000




#********** HELPER FUNCTIONS ****************

def IsCudaEnabled():

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    return device