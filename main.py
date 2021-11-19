from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment 
import gym 




unity_env = UnityEnvironment("./Builders_Sim/Builders.x86_64")
env = UnityToGymWrapper(unity_env, True,  allow_multiple_obs = True)


observation = env.reset()
for _ in range(100000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action) 
  if done:
    observation = env.reset()
