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
        
    def decay_actoion_std(self, aciton_std--lö-öö-ö)
        
        pass
    
    
    