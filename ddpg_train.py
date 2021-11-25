from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import gym
from collections import deque
import time
import os
import torch
import numpy as np

from datetime import datetime
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
  observation_next, reward, done, info = env.step(action)
  
  ddpg_agent.step(observation, action, reward, observation_next, done)
  
  observation = observation_next
  
  if done:
    observation = env.reset()


env.close()


def train():
    print("=====================================================================")

    max_ep_len = 1500

    max_training_timesteps = int(3e7)  # break training loop if timeteps > max_training_timesteps

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.01  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2e4)  # action_std decay frequency (in num timesteps)

    update_timestep = 500  # update policy every n timesteps
    K_epochs = 10  # update policy for K epochs in one PPO update

    buffer_size = 1e6
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(width=300, height=300, time_scale=10.0)

    unity_env = UnityEnvironment("./Builders_Sim/Builders_Sim.x86_64", side_channels=[channel])
    env = UnityToGymWrapper(unity_env, True)

    observation = env.reset()

    print(f"Visual Observation space is {env.observation_space.shape[0]} by {env.observation_space.shape[1]}")
    print(env.observation_space.shape[2])

    squared_image_size = env.observation_space.shape[0]

    action_size = env.action_space.shape[0]
    
    ddpg_agent = ContVisualDdpgAgent(squared_image_size, action_size, buffer_size, 40, 4, gamma)
    
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    i_time_step = 0
    i_episode = 0
    reward_window = deque(maxlen=50)
    
    while i_time_step <= max_training_timesteps:
        observation = env.reset()
        episode_reward = 0

        for i_action in range(max_ep_len):

            action = ddpg_agent.choose_action(observation, randomize = i_episode < 300)
            
            
            observation_next, reward, done, info = env.step(action)

          
            
            ddpg_agent.step(observation, action, reward, observation_next, done)
                        
            observation = observation_next
            
            
            
            print('\rEpisode {} Step {} \tavg Score: {:.2f}'.format(i_episode, i_time_step, np.mean(reward_window)),
                  end="")
                

            if done:          
                break
              
        i_episode += 1
        observation = env.reset()
        
        reward_window.append(episode_reward)
        if i_episode % 50 == 0:
                print('\rEpisode {} Step {} \tavg Score: {:.2f}'.format(i_episode, i_time_step, np.mean(reward_window)))
                ddpg_agent.save('weights/model.pth')

    env.close()
      
  


if __name__ == "__main__":
    train()