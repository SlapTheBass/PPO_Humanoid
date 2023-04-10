import gym
import pybullet_envs

ENV_ID = "HumanoidBulletEnv-v0"
env = gym.make(ENV_ID, render = True)

def PrintEnvironmentInfo():
    actions = env.action_space
    obs = env. observation_space

    print("Action Space: ", actions)
    print("Observation Space: ", obs)
    print(env)


def GetMainEnvironment():
    return env

def MakeTestEnvironment():
    test_env = gym.make(ENV_ID)
    return test_env

if __name__ == "__main__":
    PrintEnvironmentInfo()
