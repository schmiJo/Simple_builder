from numpy.random.mtrand import random
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple

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


class VisualActorCriticNet(nn.Module):
    
    def __init__(self, squared_img_size: int, in_channels: int,action_size: int):
        """Initialize an Actor Critic Network that 
        Args:
            squared_img_size (int): [description]
            action_size (int): [description]
        """
        super().__init__()
        
        self.cnn_head_size = 0
        
        self.cnn_body = nn.Sequential(
            nn.Conv2d(in_channels, 6, 4, stride= 2 , padding= 0 ), # halves size
            nn.MaxPool2d(2,2),# halves size
            nn.Conv2d(6, 16, kernel_size= 3, stride= 1) ,# same size
            nn.MaxPool2d(kernel_size=2, stride= 2), # halves size
            nn.Conv2d(16, 30, 4, 2) # halves size
        )
        
        with torch.no_grad():   
            # shape of the input should be the following (Batch size, color channels, x, y)
            test = torch.zeros(2, 3, squared_img_size, squared_img_size)
            test_result = self.cnn_body(test)
            self.cnn_head_size =  30 * test_result.shape[2] * test_result.shape[3]
            
        
        
        # the actor body that comes after the common cnn layer
        self.actor_body = nn.Sequential(
            nn.Linear(self.cnn_head_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh() # Tanh layer used to get values between 0 and 1
        )
        
        # the critic body that comes after the common cnn layer
        self.critic = nn.Sequential(
            nn.Linear(self.cnn_head_size, 128 + action_size),
            nn.ReLU(),
            nn.Linear(128, 64), # the action gets inserted at this step to the feed forward network
            nn.ReLU(),
            nn.Linear(64, 1) # Value reduced to 1 to get an acurate value of the state
            )
    
    def forward(self):
        raise NotImplementedError
    
    
    def choose_action(self, visual_obs: torch.Tensor):
        """
        Choose an action based on the visual state given
        """    
        cnn_head = self.cnn_body(visual_obs) # shape (batch size ,channels (30), out shape 2, out shape 3)
        # flatten the cnn_head into a digesible format for the fully connected network
        
        cnn_head = cnn_head.view(-1, self.cnn_head_size)# shape (batch size, channels * out_shape^2)
        
        action = self.actor_body(cnn_head)
        
        return action
    
    def evaluate(self, visual_obs: torch.Tensor, action: torch.Tensor):
        """Evaluate a state action pair

        Args:
            visual_state (torch.Tensor): shape(batch_size, channels, x, y )
            action (torch.Tensor): shape(batch_size, action_size)
        """
        
        cnn_head = self.cnn_body(visual_obs) # shape (batch size ,channels (30), out shape 2, out shape 3)
        # flatten the cnn_head into a digesible format for the fully connected network
        
        cnn_head = cnn_head.view(-1, self.cnn_head_size)# shape (batch size, channels * out_shape^2)
        
        state_action_tensor = torch.cat((cnn_head, action), dim=1) # shape (batch_size, chanells * out_shape^2 + action_size)
        
        value = self.critic(state_action_tensor)
        
        return value
    
class ReplayBuffer:
    
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        """Stores expiences encountered during training

        Args:
            buffer_size (int): the size of the buffer max
            batch_size (int): the size of the batches that will get sampled using the sample function
        """
        self.memory = deque(maxlen=buffer_size)# internal memory
        self.batch_size = batch_size
        self.experience_tuple = namedtuple("Experience", field_names=["visual_obs", "action", "reward", "next_visual_obs", "done"])
        
        
    def add(self, visual_obs: np.array, action: np.array, reward: np.number, next_visual_obs: np.array, done: np.number ):
        """Add an experience to the replay buffer

        Args:
            visual_obs (np.array): obs stands for observation
            action (np.array): [description]
            reward (np.number): [description]
            next_visual_obs (np.array): [description]
            done (np.number): [description]
        """
        e = self.experience_tuple(visual_obs, action, reward, next_visual_obs, done)
        self.memory.append(e)   
        
    def __len__(self):
        return len(self.memory)
    
    def sample(self):
        
        exp = random.sample(self.memory, k= self.batch_size)
        
        visual_obss =  torch.from_numpy(np.array([e.visual_obs for e in exp])).float().to(device)
        actions =  torch.from_numpy(np.array([e.action for e in exp])).float().to(device)
        rewards =  torch.from_numpy(np.array([e.reward for e in exp])).float().to(device)
        next_visual_obss =  torch.from_numpy(np.array([e.next_visual_obs for e in exp])).float().to(device)
        dones =  torch.from_numpy(np.array([e.done for e in exp]).astype(np.uint8)).float().to(device)
        
        return visual_obss, actions, rewards, next_visual_obss, dones
        
        
class ContVisualDdpgAgent():
    """Continous Visual Ddpg Agent
    """
    
    
    def __init__(self, visual_obs_size: int, action_size: int, buffer_size: int, batch_size: int, visual_obs_channels: int = 3, gamma: int = 0.99, tau: int = 1e-3, lr_actor: int = 3e-4, lr_critic: int = 1e-4, lr_cnn = 2e-4) -> None:
        """
        Args:
            visual_obs_size (int): one squared lenght of the ob
            action_size (int): size of the continous action vector
            buffer_size (int): size fo the experience replay buffer
            batch_size (int): size of the batch used for learnign
            gamma (int, optional): Gamma reward decay rate. Defaults to 0.99.
            tau (int, optional): Updating Factor from the local to the target network. Defaults to 1e-3.
            lr_actor (int, optional): Learning rate of the actor. Defaults to 3e-4.
            lr_critic (int, optional): Learnign rate of the crititc (it is advised, that the critics learning rate is lower than the actor learnign rate). Defaults to 1e-4.
        """
        
        self.visual_obs_size = visual_obs_size
        self.visual_obs_channels = visual_obs_channels
        self.action_size = action_size
        self.buffer_size = int(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lr_cnn = lr_cnn
        
        
        # Replay Buffer
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
        
        self.local_net = VisualActorCriticNet(visual_obs_size, visual_obs_channels, action_size)
        
        self.target_net = VisualActorCriticNet(visual_obs_size, visual_obs_channels, action_size)
        
        self.hard_update()
        
        self.actor_optim = optim.Adam([
            {'params': self.local_net.actor_body, 'lr': lr_actor},
            {'params': self.local_net.cnn_body, 'lr': lr_actor}  
        ])
        
        
        self.critic_optim = optim.Adam([
            {'params': self.local_net.critic, 'lr': lr_critic},
            {'params': self.local_net.cnn_body, 'lr': lr_critic}  
        ])
        
        # OU Noise is not used in this implementation
        
        
    def act(self, visual_obs: np.ndarray, randomize: bool = True) -> np.ndarray:
        
        visual_obs = np.moveaxis(visual_obs, -1, 0) # get the channels in front
        visual_obs = torch.from_numpy(visual_obs).unsqueeze(dim = 0).float().to(device) # unsqueeze to accomodate for bath size and convert to torch sensor
        
        self.local_net.eval() # switch to eval mode
        
        with torch.no_grad(): # no gradients required
            # Aquire an action from the network
            action = self.local_net.choose_action(visual_obs=visual_obs).squeeze().cpu().data.numpy()
            
        self.local_net.train() # switch to train mode
    
        if randomize:
            action += np.random.normal(0, 0.3, size=(self.action_size))
        
        return action
    
    
    def step(self, visual_obs: np.ndarray, action: np.ndarray, reward: np.float32, next_visual_obs : np.ndarray, done: np.int8):
        """Add the sars pair to the memory

        Args:
            visual_obs (np.ndarray): obs stands for observation
            action (np.ndarray): action taken to get to next obs from obs
            reward (np.float32): reward gotten durign that one step
            next_visual_obs (np.ndarray): [description]
            done (np.int8): determines whether this was the end of that episode
        """
        # reshape the visual obs
        visual_obs = np.moveaxis(visual_obs, -1, 0)
        next_visual_obs = np.moveaxis(next_visual_obs, -1, 0)
        
        self.memory.add(visual_obs, action, reward, next_visual_obs, done)
        
        if len(self.memory) > self.batch_size:
            exp = self.memory.sample()
            self.learn(exp)
    
    def learn(self, exp):
        
        visual_obs, actions, rewards, next_visual_obs, dones = exp
        
        # train the criric
        next_actions = self.target_net.choose_action(visual_obs)
        # get expected values from local critic by passsing in both the states and the action
        Q_expected = self.local_net.evaluate(visual_obs, actions)
        # get the next expected Q values from the local critic by passing in both the next_obs and the next_actions
        next_Q_target = self.target_net.evaluate(next_visual_obs, next_actions)
        # Compute the Q target for the current state
        Q_targets = rewards + (self.gamma * next_Q_target * (1- dones))
        # Calculate the loss function 
        critc_loss = F.mse_loss(Q_expected, Q_targets) # the expected q values should be close to the q targets
        self.critic_optim.zero_grad()
        critc_loss.backward()
        self.critic_optim.step()
        
        # train the actor
        action_pred = self.local_net.choose_action(visual_obs)
        actor_loss = -self.local_net.critic(visual_obs, action_pred).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        
        self.soft_update()
        
        
        pass
    
    def hard_update(self):
        """Copy the weights and biases from the local to the target network"""
        for target_param, local_param in zip(self.target_net.parameters(), self.local_net.parameters()):
            target_param.data.copy_(local_param.data)
        
    def soft_update(self):
        """ Copy the weights and biases from the local to the target nework using the parameter tau"""
        for target_param, local_param in zip(self.target_net.parameters(), self.local_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + local_param.data * self.tau)
        

        
        
class ActorNet(nn.Module):
    
    def __init__(self):
        super().__init__() 