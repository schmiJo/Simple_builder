from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment 
import gym  
from ddpg_agent import Agent
from collections import deque

unity_env = UnityEnvironment("./Builders_Sim.app")
env = UnityToGymWrapper(unity_env, True)


observation = env.reset()

print(env.observation_space.shape[0]);

print(env.observation_space.shape[1]);

agent = Agent(action_size=env.action_space.shape[0], state_size=7056)

for _ in range(100000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
  
  
  
  if done:
    observation = env.reset()


env.close()