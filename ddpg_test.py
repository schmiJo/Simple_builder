from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment 
import gym  
from collections import deque
from ddpg import ContVisualDdpgAgent

unity_env = UnityEnvironment("./Builders_Sim.app")
env = UnityToGymWrapper(unity_env, True)


observation = env.reset()

print(env.observation_space.shape[0]);

print(env.observation_space.shape[1]);

ddpg_agent = ContVisualDdpgAgent(84, 5, 2e6, 20)


for _ in range(100000):
  env.render() 
  action = ddpg_agent.act(observation)
  observation, reward, done, info = env.step(action)
  
  print(action)
  
  if done:
    observation = env.reset()


env.close()