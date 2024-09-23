from collections import deque
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathery_env.wrappers.flatten_action import FlattenActionWrapper
from pathery_env.wrappers.flatten_board_observation import FlattenBoardObservationWrapper
from pathery_env.envs.pathery import PatheryEnv

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
  env = gym.make('pathery_env/Pathery-FromMapString', render_mode='ansi', map_string='17.9.13.Normal...1727150400:,r3.10,r1.4,f1.,r3.2,r1.,r1.5,r1.1,c1.,r1.2,f1.,r3.2,c2.12,f1.,r3.14,r1.,f1.,s1.1,r1.13,f1.,r3.6,r1.4,r1.3,f1.,r3.1,r1.2,r1.10,f1.,r3.2,r1.12,f1.,r3.,r1.10,r1.3,f1.')

  env = FlattenActionWrapper(env)
  # env = FlattenBoardObservationWrapper(env) # Uncomment for dense NN
  return env

class DenseDQN(nn.Module):

  def __init__(self, n_observations, n_actions):
    super(DenseDQN, self).__init__()
    print(f'Initializing net with {n_observations} observations and {n_actions} actions')
    self.dense = nn.Sequential(
      nn.Linear(n_observations, 64),
      nn.ReLU(),
      nn.Linear(64, 64),
      nn.ReLU(),
      nn.Linear(64, n_actions)
    )

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    return self.dense(x)

class ConvDQN(nn.Module):

  def __init__(self, input_channels, grid_height, grid_width, n_actions):
    super(ConvDQN, self).__init__()
    print(f'Initializing net with input channels: {input_channels}, grid size: ({grid_height}, {grid_width}), and {n_actions} actions')

    # Convolutional layers
    conv_final_channel_count = 64
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(in_channels=64, out_channels=conv_final_channel_count, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
    )

    # Calculate the size of the output from conv layers for the fully connected layer
    # Assuming the conv layers do not change the spatial dimensions (stride=1 and padding=1)
    conv_output_size = conv_final_channel_count * grid_height * grid_width

    # Fully connected layer
    self.fc = nn.Sequential(
      nn.Linear(conv_output_size, n_actions)
    )

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    if x.dim() == 3:
      # Add a batch dimension
      x = x.unsqueeze(0)

    # Pass through convolutional layers
    x = self.conv(x)

    # Flatten the output for the fully connected layer
    x = x.view(x.size(0), -1)

    # Pass through fully connected layer
    return self.fc(x)

def isWrappedBy(env, wrapper_type):
  """Recursively unwrap env to check if any wrapper is of type wrapper_type."""
  current_env = env
  while isinstance(current_env, gym.Wrapper):
    if isinstance(current_env, wrapper_type):
      return True
    current_env = current_env.env  # Unwrap to the next level
  return False

def denseFromEnv(env):
  n_observations = int(gym.spaces.utils.flatdim(env.observation_space))
  n_actions = int(env.action_space.n)
  return DenseDQN(n_observations, n_actions)

def convFromEnv(env):
  n_actions = int(env.action_space.n)
  return ConvDQN(*env.observation_space[PatheryEnv.OBSERVATION_BOARD_STR].shape, n_actions)

def observationToTensor(env, obs, device):
  """Returns the observation as a pytorch tensor on the specified device with the batch dimension added"""
  return torch.tensor(obs[PatheryEnv.OBSERVATION_BOARD_STR], dtype=torch.float32, device=device).unsqueeze(0)

def getDevice():
  return torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
  )

def select_action(env, stateTensor, policy_net, device, eps_threshold, useMask=False, deterministic=False):
  """Returns the action as a pytorch tensor on the specified device with the batch dimension added"""
  explore = False
  if not deterministic:
    explore = np.random.random() <= eps_threshold
  if explore:
    if False: # TODO: Action masking branch
      mask = stateTensor['action_mask'].flatten()
      # Sample actions until we get one which is valid according to the mask
      while True:
        action_index = env.action_space.sample()
        if mask[action_index] == 1:
          return torch.tensor([action_index], device=device, dtype=torch.long)
    else:
      # Any action sample is valid without masking
      action_index = env.action_space.sample()
      return torch.tensor([[action_index]], device=device, dtype=torch.long)
  else:
    with torch.no_grad():
      # t.max(1) will return the largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
      netResult = policy_net(stateTensor)
      if False: # TODO: Action masking branch
        # Apply action mask
        mask = torch.tensor(stateTensor["action_mask"], dtype=torch.float32, device=device)
        flattened_mask = mask.flatten().unsqueeze(0)
        masked_result = torch.where(flattened_mask == 1, netResult, torch.tensor(-float('inf')))
        return masked_result.max(1).indices.view(1,1).squeeze(0)
      else:
        return netResult.max(1).indices.view(1,1)