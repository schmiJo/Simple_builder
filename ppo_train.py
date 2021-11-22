from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import gym
from collections import deque
import time
import os
import torch
import numpy as np

from datetime import datetime
from ppo import PPO


## ################## Training #################

def train():
    print("=====================================================================")

    max_ep_len = 1500
    max_training_steps = int(3e7)

    max_training_timesteps = int(3e7)  # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
    save_model_freq = int(1e5)  # save model frequency (in num timesteps)

    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.01  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.05  # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)

    update_timestep = 300  # update policy every n timesteps
    K_epochs = 50  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    random_seed = 0  # set random seed if required (0 = no random seed)

    unity_env = UnityEnvironment("./Builders_Sim.app")
    env = UnityToGymWrapper(unity_env, True)

    observation = env.reset()

    print(f"Visual Observation space is {env.observation_space.shape[0]} by {env.observation_space.shape[1]}")
    print(env.observation_space.shape[2])

    squared_image_size = env.observation_space.shape[0]

    action_size = env.action_space.shape[0]

    ppo_agent = PPO(squared_image_size, action_size, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    i_time_step = 0
    i_episode = 0
    reward_window = deque(maxlen=1000)

    while i_time_step <= max_training_timesteps:

        observation = env.reset()
        episode_reward = 0

        for i_action in range(max_ep_len):

            action = ppo_agent.select_action(observation)

            observation, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            episode_reward += reward
            i_time_step += 1

            # update PPO agent
            if i_time_step % update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of output action distribution
            if i_time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            reward_window.append(episode_reward)
            print('\rEpisode {} Step {} \tavg Score: {:.2f}'.format(i_episode, i_time_step, np.mean(reward_window)),
                  end="")
                

            if done:          
                break
              
        i_episode += 1
        
        if i_episode % 2 == 0:
                print('\rEpisode {} Step {} \tavg Score: {:.2f}'.format(i_episode, i_time_step, np.mean(reward_window)))
                ppo_agent.save('weights/model.pth')

    env.close()

if __name__ == "__main__":
    train()
