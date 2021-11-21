from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment 
import gym  
from collections import deque

unity_env = UnityEnvironment("./Builders_Sim.app")
env = UnityToGymWrapper(unity_env, True)


observation = env.reset()

print(env.observation_space.shape[0]);

print(env.observation_space.shape[1]);


for _ in range(100000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  
  if done:
    observation = env.reset()


env.close()