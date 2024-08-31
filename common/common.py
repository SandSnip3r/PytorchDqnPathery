from collections import deque
import gymnasium as gym
import numpy as np
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
  env = gym.make('pathery_env/Pathery-v0', render_mode='ansi')

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

  return FlattenActionWrapper(env)

# Define class DQN
class DQN(nn.Module):

  def __init__(self, n_observations, n_actions):
    super(DQN, self).__init__()
    print(f'Initializing net with {n_observations} observations and {n_actions} actions')
    self.layer1 = nn.Linear(n_observations, 1024)
    self.layer2 = nn.Linear(1024, 512)
    self.layer3 = nn.Linear(512, n_actions)

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)