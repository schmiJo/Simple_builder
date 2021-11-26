import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

#Region Set device
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
        self.visual_obss = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.visual_obss[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        
    def add(self, action, visual_obs, logprob, reward, is_terminal):
        self.actions.append(action)
        self.visual_obss.append(visual_obs)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(is_terminal)
        
        
class VisualActorCritric(nn.Module):
    
    def __init__(self, square_image_size: int, action_size: int, action_std_init: int):
        super().__init__()
        
        self.action_size = action_size
        
        self.action_var = torch.full((action_size, ), action_std_init ** 2 ).to(device)
        
        print(self.action_var.shape)
        
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
            test = torch.zeros((2, 3, square_image_size, square_image_size))
            result = self.cnn_head(test)
            self.cnn_head_size = 30 * result.shape[2]*result.shape[3]
             

        # actor
        self.actor = nn.Sequential(
                        nn.Linear(self.cnn_head_size, 64),  
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_size),
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
        self.action_var = torch.full((self.action_size, ), new_action_std ** 2 ).to(device)
        
    def forward(self):
        raise NotImplementedError
        
        
    def act(self, visual_obs: torch.Tensor):
        """Generate an action given a state

        Args:
            visual_obs (torch.Tensor): Visual Observation of the agent: shape(batch_size, channel_size (3), width (84), height(84ch))
        """
        
        cnn_res = self.cnn_head(visual_obs).view(-1, self.cnn_head_size) # first the state gets passed to the cnn network and flattened into the shape (batch_size, cnn_head_size)
        
        action_mean = self.actor(cnn_res) # forward it to the actor net to generate the mean of the action that will be taken shape(batch_size, action_size)
        cov_matrix = torch.diag(self.action_var).unsqueeze(dim = 0) # since the same variance is desired in all directions of the action_space, the variance is simply expanded to fill the same shape as the action mean shape(batch_size, aciton_size)
        dist = MultivariateNormal(action_mean, cov_matrix) # use the multivariable normal distribution to sample from, mean is the action vector gotten from the action net and the variance is the computed covariance matric
        action = dist.sample()
        action_logprobs = dist.log_prob(action)
        
        return action.detach(), action_logprobs.detach()
    
    def evaluate(self, visual_obs: torch.Tensor, action):
        
        cnn_res = self.cnn_head(visual_obs).view(-1, self.cnn_head_size)
        
        action_mean = self.actor(cnn_res)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        
        dist_entropy = dist.entropy()
        # Todo: Try using the cnn_res from above and see how the gradients check out
        cnn_res = self.cnn_head(visual_obs).view(-1, self.cnn_head_size) 
        state_values = self.critic(cnn_res)
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, squared_img_size: int, action_size: int, lr = 1e-4 , gamma = 0.99, K_epochs = 5 , eps_clip = 0.1, action_std_init = 0.3) -> None:
        
        self.action_std = action_std_init
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.squared_img_size = squared_img_size
        self.action_size = action_size
        self.lr = lr
        self.K_epochs = K_epochs
        
        
        self.memory = RolloutBuffer()
        
        self.policy = VisualActorCritric(squared_img_size, action_size, self.action_std).to(device)
        
        self.optim = torch.optim.Adam([
            {'params': self.policy.cnn_head.parameters(), 'lr': lr},
            {'params': self.policy.actor.parameters(), 'lr' : lr },
            {'params': self.policy.critic.parameters(), 'lr': lr }
        ])
        
        self.policy_old = VisualActorCritric(squared_img_size, action_size, self.action_std)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        
    def decay_actoion_std(self, action_std_decay_rate, min_action_std):
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
        pass
    
    def choose_action(self, visual_obs):
        #visual_obs = np.moveaxis(visual_obs,-1, 0) do this 
        with torch.no_grad():
            visual_obs: torch.Tensor = torch.from_numpy(visual_obs).float().to(device).unsqueeze(dim= 0)
            action, action_logprob = self.policy_old.act(visual_obs)
            
        return action, action_logprob
    
    def update(self):
        """Updates and learns from the experiences in the buffer, clears the buffer afterwards
        """
        
        # Monte carlo estimate
        rewards = []
        discounted_reward = 0
        for rewards, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = rewards + self.gamma * discounted_reward
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.memory.visual_obss, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.memory.logprobs, dim=0)).detach().to(device)
        
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
            self.optim.zero_grad()
            loss.mean().backward()
            self.optim.step()
            
        # Copy new weights into old polcy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        
        # clean buffer
        self.memory.clear()
        
        pass
            
    def step(self, action, visual_obs, action_logprob, reward, is_terminal):
        """
        Add to the buffer the relevant information
        Args:
            action ([type]): [description]
            visual_obs ([type]): [description]
            action_logprob ([type]): [description]
            reward ([type]): [description]
            is_terminal (bool): [description]
        """
        self.memory.add(action, visual_obs, action_logprob, reward, is_terminal)
        
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
        pass

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
            
    
    