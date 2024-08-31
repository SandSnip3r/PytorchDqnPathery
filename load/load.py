import gymnasium as gym
import pathery_env
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from torch.utils.tensorboard import SummaryWriter
from common import common

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def main(model_path):
  env = common.getEnv()

  # if GPU is to be used
  device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
  )
  print(f'Using device {device}')

  # Get number of actions from gym action space
  n_actions = env.action_space.n
  # Get the number of state observations
  n_observations = np.sum(env.observation_space.nvec)

  def observationToTensor(obs):
    # Pull observation out of observation & mask dict
    flattened = gym.spaces.utils.flatten(env.observation_space, obs)
    return torch.tensor(flattened, dtype=torch.float32, device=device).unsqueeze(0)

  policy_net = common.DQN(n_observations, n_actions).to(device)
  print(f'Policy net: {policy_net}')
  policy_net.load_state_dict(torch.load(model_path, weights_only=True))
  print(f'Policy net: {policy_net}')

  observation, info = env.reset()
  print(env.render())
  print(f'Initial info: {info}')

  done = False

  while not done:
    action_values = policy_net(observationToTensor(observation))
    max_action = action_values.max(1).indices.view(1,1).squeeze(0)

    observation, reward, terminated, truncated, info = env.step(max_action.item())
    print(env.render())
    print(f'info: {info}')

    done = terminated

if __name__ == "__main__":
  main()