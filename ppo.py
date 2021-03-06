import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np


################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class VisualActorCritic(nn.Module):
    def __init__(self, squared_image_side, action_dim, action_std_init):
        super(VisualActorCritic, self).__init__()

        self.action_dim = action_dim
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        self.cnn_head_size = 0
        
        self.cnn_head = nn.Sequential(
            nn.Conv2d(3, 6, 4),
            nn.MaxPool2d(4,4),
            nn.Conv2d(6, 16, 4),
            nn.MaxPool2d(3,3),
            nn.Conv2d(16, 30, 4 )
        )
        
        
        with torch.no_grad():
            # Batch size, color channels, x , y 
            test = torch.zeros((2, 3, squared_image_side, squared_image_side))
            result = self.cnn_head(test)
            self.cnn_head_size = 30 * result.shape[2]*result.shape[3]
             

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(self.cnn_head_size, 64),  
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_dim),
                        nn.Tanh()
                    )
           
        
        # critic
        self.critic = nn.Sequential(
                        nn.Linear(self.cnn_head_size, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)


    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        # The expected shape of the state is (batch size, channelsize (3), width (84), height(84) ) 
        
        cnn_res = self.cnn_head(state).view(-1, self.cnn_head_size)
        
        action_mean = self.actor(cnn_res)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):
        # The expected shape of the state is (batch size, channelsize (3), width (84), height(84) ) 
        cnn_res = self.cnn_head(state).view(-1, self.cnn_head_size) 
        
        action_mean = self.actor(cnn_res)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        print("evaluation covariance matrix")
        print(cov_mat)
        print(cov_mat.shape)
        
        # For Single Action Environments.
        if self.action_dim == 1:
            action = action.reshape(-1, self.action_dim)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        cnn_res = self.cnn_head(state).view(-1, self.cnn_head_size) 
        state_values = self.critic(cnn_res)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init=0.6):

        self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = VisualActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.cnn_head.parameters(), 'lr': lr_actor},
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = VisualActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def set_action_std(self, new_action_std):
        
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)
        

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if (self.action_std <= min_action_std):
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state):

        with torch.no_grad():
            state: torch.Tensor = torch.from_numpy(np.moveaxis(state, -1, 0)).float().to(device).unsqueeze(dim=0)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.states.append(state.squeeze())
        self.buffer.actions.append(action.squeeze())
        self.buffer.logprobs.append(action_logprob)
       

        return action.detach().cpu().numpy().flatten()

    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        
        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        pass

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
       






