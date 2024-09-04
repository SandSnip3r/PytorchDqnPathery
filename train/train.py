import gymnasium as gym
import pathery_env
import math
import random
import numpy as np
from collections import namedtuple, deque
from itertools import count
from torch.utils.tensorboard import SummaryWriter
import torch.profiler
from common import common

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args):
    """Save a transition"""
    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

def main():
  env = common.getEnv()
  device = common.getDevice()
  print(f'Using device {device}')

  # BATCH_SIZE is the number of transitions sampled from the replay buffer
  # GAMMA is the discount factor as mentioned in the previous section
  # EXPLORATION_INITIAL_EPS is the starting value of epsilon
  # EXPLORATION_FINAL_EPS is the final value of epsilon
  # EXPLORATION_FRACTION specifies at what point in training does exploration reach the final value
  # TAU is the update rate of the target network
  # LEARNING_RATE is the learning rate of the ``AdamW`` optimizer
  BATCH_SIZE = 256
  GAMMA = 0.99
  EXPLORATION_INITIAL_EPS = 1.0
  EXPLORATION_FINAL_EPS = 0.05
  EXPLORATION_FRACTION = 0.1
  TAU = 1 # 0.005
  LEARNING_RATE = 1e-4
  # Number of actions per copy of policy->target
  TARGET_UPDATE_INTERVAL = 10000
  # Number of actions to take per optimize_model() call
  TRAIN_FREQUENCY = 4
  # Sample count for statistics
  RUNNING_AVERAGE_LENGTH = 32
  # Number of actions per evalutation
  EVAL_FREQUENCY = 1000

  policy_net = torch.jit.script(common.convFromEnv(env).to(device))
  target_net = torch.jit.script(common.convFromEnv(env).to(device))
  print(f'Policy net: {policy_net}')
  target_net.load_state_dict(policy_net.state_dict())

  optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
  memory = ReplayMemory(100000)

  writer = SummaryWriter()

  def calculateExplorationEpsilon(action_index, total_action_count):
    progress = action_index / total_action_count
    return EXPLORATION_INITIAL_EPS + (EXPLORATION_FINAL_EPS-EXPLORATION_INITIAL_EPS) * (min(progress, EXPLORATION_FRACTION) / EXPLORATION_FRACTION)

  def optimize_model():
    if len(memory) < BATCH_SIZE:
      return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    if all(s is None for s in batch.next_state):
      # We require at least one next action for any model training
      return

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([common.observationToTensor(s, device).unsqueeze(0) for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat([common.observationToTensor(s, device).unsqueeze(0) for s in batch.state])
    action_batch = torch.cat(batch.action).unsqueeze(1)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
      next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

  def evalModel():
    done = False
    observation, info = env.reset()
    episodeReward = 0
    stepCount = 0
    while not done:
      action = common.select_action(env, observation, policy_net, device, eps_threshold=None, deterministic=True)
      observation, reward, terminated, truncated, _ = env.step(action.item())
      episodeReward += reward
      stepCount += 1
      done = terminated
    return stepCount, episodeReward

  total_action_count = 300_000

  episodeRewardRunningAverage = common.RunningAverage(RUNNING_AVERAGE_LENGTH)
  episodeLengthRunningAverage = common.RunningAverage(RUNNING_AVERAGE_LENGTH)

  # Initialize the environment and get its state
  state, info = env.reset()
  episodeReward = 0
  episodeStepIndex = 0
  episode_index = 0
  needToEval = False
  for action_index in range(total_action_count):
    eps_threshold = calculateExplorationEpsilon(action_index, total_action_count)
    writer.add_scalar("Epsilon", eps_threshold, action_index)

    action = common.select_action(env, state, policy_net, device, eps_threshold)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated
    episodeReward += reward

    if terminated:
      next_state = None
    else:
      next_state = observation

    # Store the transition in memory
    memory.push(state, action, next_state, reward)

    # Move to the next state
    state = next_state

    if (action_index+1) % TRAIN_FREQUENCY == 0:
      # Perform one step of the optimization (on the policy network)
      optimize_model()

    if (action_index+1) % TARGET_UPDATE_INTERVAL == 0:
      print(f'Updating target network (episode #{episode_index}, action #{action_index})')
      # Soft update of the target network's weights
      # θ′ ← τ θ + (1 −τ )θ′
      target_net_state_dict = target_net.state_dict()
      policy_net_state_dict = policy_net.state_dict()
      for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
      target_net.load_state_dict(target_net_state_dict)

    if (action_index+1) % EVAL_FREQUENCY == 0:
      needToEval = True

    if done:
      if needToEval:
        length, reward = evalModel()
        episodeLengthRunningAverage.add(length)
        episodeRewardRunningAverage.add(reward)
        writer.add_scalar("Episode_reward", episodeRewardRunningAverage.average(), action_index)
        writer.add_scalar("Episode_length", episodeLengthRunningAverage.average(), action_index)
        needToEval = False
      if (episode_index+1) % 100 == 0:
        print(f'Episode {episode_index} complete')
      episode_index += 1
      episodeReward = 0
      episodeStepIndex = 0
      state, info = env.reset()
      continue

    episodeStepIndex += 1

  policy_net.save('policy_net_script.pt')

if __name__ == "__main__":
  main()