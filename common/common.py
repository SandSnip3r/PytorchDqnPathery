from collections import deque
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class RunningAverage:
  def __init__(self, size):
    self.size = size
    self.buffer = deque(maxlen=size)
    self.total = 0

  def add(self, value):
    if len(self.buffer) == self.size:
      # Remove the oldest element and subtract it from the total
      oldest = self.buffer.popleft()
      self.total -= oldest
    # Add the new value and update the total
    self.buffer.append(value)
    self.total += value

  def average(self):
    if not self.buffer:
      return 0
    return self.total / len(self.buffer)

def getEnv():
  env = gym.make('pathery_env/Pathery-FromMapString', render_mode='ansi', map_string='17.9.14.Normal...1725422400:,r3.8,r1.6,f1.,r3.15,f1.,s1.3,r1.5,r1.5,f1.,r3.15,f1.,r3.8,c2.6,f1.,r3.3,r1.4,r1.6,f1.,r3.,r1.8,c1.5,f1.,r3.2,r1.3,r1.,r1.7,f1.,r3.1,r1.13,f1.')

  class FlattenActionWrapper(gym.Wrapper):
    def __init__(self, env):
      super(FlattenActionWrapper, self).__init__(env)
      self.original_action_space = env.action_space
      self.action_space = gym.spaces.Discrete(np.prod(env.action_space.nvec))
    
    def step(self, action):
      # Convert flattened action back to multi-discrete
      unflattened_action = np.unravel_index(action, self.original_action_space.nvec)
      return self.env.step(unflattened_action)
    
    def reset(self, **kwargs):
      # Pass any arguments (e.g., seed) to the underlying environment's reset method
      return self.env.reset(**kwargs)

  class ConvObservationWrapper(gym.Wrapper):
    def __init__(self, env):
      super(ConvObservationWrapper, self).__init__(env)
      self.original_observation_space = env.observation_space['board']
      self.channel_count = self.original_observation_space[0][0].n
      self.height = self.original_observation_space.shape[0]
      self.width = self.original_observation_space.shape[1]

    def obsToImage(self, obs):
      board = obs['board']
      oneHot = np.zeros((self.channel_count, board.shape[0], board.shape[1]), dtype=np.float32)
      for i in range(self.channel_count):
        oneHot[i] = (board == i)
      return {
        'board': oneHot,
        'action_mask': obs['action_mask']
      }

    def step(self, action):
      obs, *other = self.env.step(action)
      return self.obsToImage(obs), *other

    def reset(self, **kwargs):
      # Pass any arguments (e.g., seed) to the underlying environment's reset method
      obs, *other = self.env.reset(**kwargs)
      return self.obsToImage(obs), *other

  env = FlattenActionWrapper(env)
  return ConvObservationWrapper(env)

class DenseDQN(nn.Module):

  def __init__(self, n_observations, n_actions):
    super(DenseDQN, self).__init__()
    print(f'Initializing net with {n_observations} observations and {n_actions} actions')
    self.layer1 = nn.Linear(n_observations, 512)
    self.layer2 = nn.Linear(512, 512)
    self.layer3 = nn.Linear(512, n_actions)

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)

class ConvDQN(nn.Module):

  def __init__(self, input_channels, grid_height, grid_width, n_actions):
    super(ConvDQN, self).__init__()
    print(f'Initializing net with input channels: {input_channels}, grid size: ({grid_height}, {grid_width}), and {n_actions} actions')

    # Convolutional layers
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

    # Calculate the size of the output from conv layers for the fully connected layer
    # Assuming the conv layers do not change the spatial dimensions (stride=1 and padding=1)
    conv_output_size = 64 * grid_height * grid_width

    # Fully connected layer
    self.fc = nn.Linear(conv_output_size, n_actions)

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    if x.dim() == 3:
      # Add a batch dimension
      x = x.unsqueeze(0)

    # Pass through convolutional layers
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))

    # Flatten the output for the fully connected layer
    x = x.view(x.size(0), -1)

    # Pass through fully connected layer
    return self.fc(x)

def convFromEnv(env):
  n_actions = int(env.action_space.n)
  return ConvDQN(int(env.channel_count), env.height, env.width, n_actions)

def observationToTensor(obs, device):
  # Pull observation out of observation & mask dict
  # flattened = gym.spaces.utils.flatten(env.observation_space, obs)
  # return torch.tensor(flattened, dtype=torch.float32, device=device).unsqueeze(0)
  return torch.tensor(obs['board'], dtype=torch.float32, device=device)

def getDevice():
  return torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
  )

def select_action(env, state, policy_net, device, eps_threshold, deterministic=False):
  explore = False
  if not deterministic:
    explore = random.random() <= eps_threshold
  if explore:
    mask = state['action_mask'].flatten()
    # Sample actions until we get one which is valid according to the mask
    while True:
      action_index = env.action_space.sample()
      if mask[action_index] == 1:
        return torch.tensor([action_index], device=device, dtype=torch.long)
  else:
    with torch.no_grad():
      # t.max(1) will return the largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
      observationAsTensor = observationToTensor(state, device)
      netResult = policy_net(observationAsTensor)
      # Apply action mask
      mask = torch.tensor(state["action_mask"], dtype=torch.float32, device=device)
      flattened_mask = mask.flatten().unsqueeze(0)
      masked_result = torch.where(flattened_mask == 1, netResult, torch.tensor(-float('inf')))
      return masked_result.max(1).indices.view(1,1).squeeze(0)