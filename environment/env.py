import gym
import pybullet_envs

ENV_ID = "HumanoidBulletEnv-v0"
render = True
env = gym.make(ENV_ID, renders = render)

def PrintEnvironmentInfo():
    actions = env.action_space
    obs = env. observation_space

    print("Action Space: ", actions)
    print("Observation Space: ", obs)
    print(env)

if __name__ == "__main__":
    PrintEnvironmentInfo()
