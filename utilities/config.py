import os
import torch

ROOT_DIR = os.path.abspath(os.curdir)

#*********** SAVE/LOAD paths *****************

SAVE_PATH = "saves"
RECORD_PATH = "records"

#****** NEURAL NETWORK PARAMETERS ************

HIDDEN_LAYERS_SIZE = 512

#********** HYPER PARAMETERS *****************

GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 32

TEST_ITERS = 100000


#********* REPLAY PARAMETERS ****************

STEPS = 100000

#********** HELPER FUNCTIONS ****************

def IsCudaEnabled():

    device = "cpu"

    if torch.cuda.is_available():
        device = "cuda"

    return device

def GetFolderCnt(path):
    folder_count = 0

    for item in os.listdir(path):
            if os.path.isdir(os.path.join(path, item)):
                folder_count += 1

    print("Liczba folderów: ", folder_count)

    return folder_count
    

def CreateRecordPath():

    path = os.path.join(ROOT_DIR, RECORD_PATH)

    if not os.path.isdir(os.path.join(path)):
        try:
            os.mkdir(path)
        except OSError:
            print(OSError)
        
    folder_count = GetFolderCnt(path)

    record_dir = "PPO_humanoid_record_" + str(folder_count)

    path = os.path.join(ROOT_DIR, RECORD_PATH, record_dir)

    if not os.path.isdir(os.path.join(path)):
        print("Your record will be saved in: \n")
        print(path + "\n\n")
        return path
    else:
        print("Record path corrupted!!!")

