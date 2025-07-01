import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from config_RL import *


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model.
    
    Args:
        state_dim: Dimension of the state input.
        action_dim: Number of possible actions.
        
    Returns:
        nn.Module: A neural network model that outputs Q-values for each action.
    """
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)  
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        return self.fc3(x)

class ReplayBuffer:
    """
    Experience Replay Buffer for a single agent.
    
    Args:
        capacity: Maximum number of experiences to store.
        
    Attributes:
        buffer (deque): A deque storing the experiences.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Push a new experience (transition) into the replay buffer.

        Args:
            state: The current state of the environment.
            action: The action taken by the agent.
            reward: The reward received after taking the action.
            next_state: The state of the environment after the action.
            done: A flag indicating whether the episode has terminated.
        """
        if isinstance(state, torch.Tensor):
            state = state.clone().detach().to('cpu')
        else:
            state = torch.tensor(state, dtype=torch.float32).to('cpu')
            
        if isinstance(action, torch.Tensor):
            action = action.clone().detach().to('cpu')
        else:
            action = torch.tensor(action, dtype=torch.long).to('cpu')
            
        if isinstance(reward, torch.Tensor):
            reward = reward.clone().detach().to('cpu')
        else:
            reward = torch.tensor(reward, dtype=torch.float32).to('cpu')
            
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.clone().detach().to('cpu')
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32).to('cpu')
            
        if isinstance(done, torch.Tensor):
            done = done.clone().detach().to('cpu')
        else:
            done = torch.tensor(done, dtype=torch.float32).to('cpu')
            
        self.buffer.append((state, action, reward, next_state, done))

          
    def sample(self, batch_size):
        """
        Randomly sample a batch of experiences from the buffer.
        
        Args:
            batch_size (int): The number of experiences to sample.
            
        Returns:
            Tuple[torch.Tensor]: Batch of (states, actions, rewards, next_states, dones).
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.tensor(actions, dtype=torch.long).unsqueeze(1),  # 단일 행동 유지
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1)
        )  

    def __len__(self):
        return len(self.buffer)
    
GLOBAL_REPLAY_BUFFER = ReplayBuffer(capacity=100000)

class DQNAgent:
    """
    DQNAgent for training a Deep Q-Network using experience replay.
    
    Args:
        state_dim: Dimension of the state space.
        action_dim: Number of possible actions.
        buffer_size: Size of the replay buffer.
        lr: Learning rate for the optimizer.
        gamma: Discount factor.
        target_update_interval: Steps between target network updates.
        device: Device to run the computations on (CPU or GPU).
        
    Attributes:
        q_network (DQN): The main Q-network.
        target_network (DQN): The target Q-network.
        optimizer: Optimizer for the Q-network.
        buffer: Replay buffer for storing experiences.
    """
    def __init__(self, state_dim, action_dim, buffer_size, lr=1e-3, gamma=0.99, target_update_interval=1000, device=DEVICE):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.update_step = 0
        self.loss = None
        self.device = device

        self.buffer = GLOBAL_REPLAY_BUFFER
        
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval().to(self.device)  

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

    def select_action(self, state, epsilon = max(0.1, 1.0 - N_EPISODES/500)):
        """
        Select an action using an epsilon-greedy policy.
        
        Args:
            state: The current state.
            epsilon: Probability of choosing a random action.
            
        Returns:
            torch.Tensor: The selected action (as a tensor on the specified device).
        """
        if isinstance(state, torch.Tensor):
            state_tensor = state.unsqueeze(0) if state.dim() == 1 else state
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = state_tensor.to(self.device)
        if random.random() < epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
            return torch.tensor(action).to(self.device)

    def update(self, batch_size):
        """
        Update the Q-network using experiences sampled from the replay buffer.
        
        Args:
            batch_size: Number of experiences to sample for the update.
        """
        if len(self.buffer) < batch_size:
            return  

        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q-values for the selected actions
        q_values = self.q_network(states).gather(1, actions)

         # Compute target Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Calculate loss between current and target Q-values
        loss = nn.MSELoss()(q_values, target_q_values)           
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.loss = loss.item()
        self.update_step += 1
        
        # Periodically update the target network to match the Q-network
        if self.update_step % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())