
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler
import imageio
import typing
import matplotlib.pyplot as plt
import os
from io import BytesIO

from PIL import Image
from collections import namedtuple, deque
from common import common
from common import Visualizer
from torch.utils.tensorboard import SummaryWriter
from grokfast_pytorch import GrokFastAdamW
from pathery_env.envs.pathery import PatheryEnv
from pathery_env.envs.pathery import CellType
from matplotlib.patches import Rectangle

from common import PrioritizedExperienceReplay
from cpp_modules import prioritized_buffer

class Transition(typing.NamedTuple):
  state: int
  action: int
  nextState: int
  reward: float

class ReplayMemory(object):

  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, item: Transition):
    """Save a transition"""
    self.memory.append(item)

  def sample(self, batch_size):
    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)

def set_seed(seed, env, determinism=False):
  # For Python's random module
  random.seed(seed)

  # For NumPy's random module
  np.random.seed(seed)

  # For PyTorch (CPU)
  torch.manual_seed(seed)

  # For PyTorch (GPU)
  if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU

  # Seed the environment
  env.action_space.seed(seed)
  env.reset(seed=seed)

  if determinism:
    # Ensure deterministic behavior in PyTorch (optional, but can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

def getStateSamples(env, stateCount, device):
  """Collects a set of states via random actions for later evaluation."""
  actionSpace = env.action_space
  observation, _ = env.reset()
  result = [common.observationToTensor(env, observation, device)]
  for i in range(stateCount-1):
    action = actionSpace.sample()
    observation, _, terminated, truncated, _ = env.step(action)
    result.append(common.observationToTensor(env, observation, device))
    if terminated or truncated:
      # Since we're not usually working with random environments, do not add the starting state again
      env.reset()

  return torch.cat(result)

def main(experimentName: str = None):
  device = common.getDevice()
  print(f'Using device {device}')

  env = common.getEnv()

  # Seed the random number generation
  INITIAL_SEED = 123
  set_seed(INITIAL_SEED, env)

  # Good parameters:
  #   Simple:
  #     Exploration fraction 0.02
  #     Action count: 1m
  #     Eval freq: 1k
  #     Network: 128,128,64 conv, followed by single dense
  # BATCH_SIZE is the number of transitions sampled from the replay buffer
  # GAMMA is the discount factor
  # EXPLORATION_INITIAL_EPS is the starting value of epsilon
  # EXPLORATION_FINAL_EPS is the final value of epsilon
  # EXPLORATION_FRACTION specifies at what point in training exploration reaches the final value
  # TAU is the update rate of the target network
  # LEARNING_RATE is the learning rate of the ``AdamW`` optimizer
  # TARGET_UPDATE_INTERVAL is the number of actions per copy of policy->target
  # TRAIN_FREQUENCY is the number of actions to take per optimize_model() call
  # RUNNING_AVERAGE_LENGTH is the sample count for statistics
  # EVAL_FREQUENCY is the number of actions per evalutation
  BATCH_SIZE = 64
  STEP_BEFORE_TRAINING = 8192
  GAMMA = 0.999
  EXPLORATION_INITIAL_EPS = 1.0
  EXPLORATION_FINAL_EPS = 0.05
  EXPLORATION_FRACTION = 0.2
  # EXPLORATION_FRACTION = 0.04 # Normal
  # EXPLORATION_FRACTION = 0.1 # Complex
  TAU = 0.95 # 0.005
  LEARNING_RATE = 1e-4
  TARGET_UPDATE_INTERVAL = 16384
  TRAIN_FREQUENCY = 4
  RUNNING_AVERAGE_LENGTH = 128
  EVAL_FREQUENCY = 1000
  RENDER_FREQUENCY = 1024
  RENDER_FPS = 60
  STATE_SAMPLE_COUNT = 512
  DOUBLE_DQN = True
  TOTAL_ACTION_COUNT = 2_000_000
  MEMORY_CAPACITY = 100_000
  PRIORITIZED_EXPERIENCE_REPLAY_ALPHA = 0.7
  PRIORITIZED_EXPERIENCE_REPLAY_BETA_INITIAL = 0.5
  PRIORITIZED_EXPERIENCE_REPLAY_BETA_FINAL = 1.0
  USE_PRIORITIZED_EXPERIENCE_REPLAY = True

  policy_net = torch.jit.script(common.convFromEnv(env).to(device))
  target_net = torch.jit.script(common.convFromEnv(env).to(device))
  print(f'Policy net: {policy_net}')
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  # optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
  optimizer = GrokFastAdamW(policy_net.parameters(), lr=LEARNING_RATE)
  if USE_PRIORITIZED_EXPERIENCE_REPLAY:
    memory = prioritized_buffer.PrioritizedExperienceReplayBufferObject(MEMORY_CAPACITY, BATCH_SIZE, PRIORITIZED_EXPERIENCE_REPLAY_ALPHA)
  else:
    memory = ReplayMemory(MEMORY_CAPACITY)

  stateSamples = getStateSamples(env, STATE_SAMPLE_COUNT, device)

  if experimentName is not None:
    writer = SummaryWriter(log_dir=os.path.join('runs', experimentName))
  else:
    writer = SummaryWriter()
  visualizer = Visualizer(RENDER_FPS)

  def calculateExplorationEpsilon(action_index):
    progress = action_index / TOTAL_ACTION_COUNT
    return EXPLORATION_INITIAL_EPS + (EXPLORATION_FINAL_EPS-EXPLORATION_INITIAL_EPS) * (min(progress, EXPLORATION_FRACTION) / EXPLORATION_FRACTION)

  def optimize_model(actionIndex, totalActionCount):
    if len(memory) < max(BATCH_SIZE, STEP_BEFORE_TRAINING):
      return
    # Get a list of Transitions
    if USE_PRIORITIZED_EXPERIENCE_REPLAY:
      percent = actionIndex/totalActionCount
      beta = PRIORITIZED_EXPERIENCE_REPLAY_BETA_INITIAL + (PRIORITIZED_EXPERIENCE_REPLAY_BETA_FINAL-PRIORITIZED_EXPERIENCE_REPLAY_BETA_INITIAL)*percent
      transitionSamples = memory.sample(beta)
      transitions = [x.item for x in transitionSamples]
      itemIds = [x.itemId for x in transitionSamples]
      importanceSamplingWeights = [x.weight for x in transitionSamples]
    else:
      transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays (specifically, they're tuples).
    batch = Transition(*zip(*transitions))

    if all(s is None for s in batch.nextState):
      # We require at least one next action for any model training
      return

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: [s is not None],
                                          batch.nextState)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.nextState if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros((BATCH_SIZE,1), device=device)
    with torch.no_grad():
      if DOUBLE_DQN:
        # Double DQN
        # Action selection using the online network (policy_net)
        policy_net.eval()
        next_state_actions = policy_net(non_final_next_states).max(1).indices
        # Action evaluation using the target network (target_net)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
      else:
        # Traditional DQN
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    if USE_PRIORITIZED_EXPERIENCE_REPLAY:
      criterion = nn.SmoothL1Loss(reduction='none')
      losses = criterion(state_action_values, expected_state_action_values)
      isWeightsTensor = torch.tensor(importanceSamplingWeights, dtype=torch.float32, device=device)
      weightedLosses = losses.squeeze() * isWeightsTensor
      loss = weightedLosses.mean()
      # Update priorities of Transitions we just trained on with their new losses
      with torch.no_grad():
        errors = torch.abs(expected_state_action_values - state_action_values).squeeze().cpu()
        if len(itemIds) != len(errors):
          raise ValueError(f'Expecting number of itemIds ({len(itemIds)}) and errors ({len(errors)}) to be the same')
        for index, itemId in enumerate(itemIds):
          memory.updatePriority(itemId, errors[index])
    else:
      # Compute Huber loss
      criterion = nn.SmoothL1Loss()
      loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

  def evalModel(env, policy_net, stateSamples, action_index, writer, device):
    policy_net.eval()
    # Deterministically use the model to play one episode. Log the length & reward.
    done = False
    observation, info = env.reset()
    observationTensor = common.observationToTensor(env, observation, device)
    episodeReward = 0
    stepCount = 0
    while not done:
      actionTensor = common.select_action(env, observationTensor, policy_net, device, eps_threshold=None, deterministic=True)
      observation, reward, terminated, truncated, _ = env.step(actionTensor.item())
      observationTensor = common.observationToTensor(env, observation, device)
      episodeReward += reward
      stepCount += 1
      done = terminated

    finalPathLength = len(env.unwrapped.currentPath)
    writer.add_scalar("eval/final_path_length", finalPathLength, action_index)
    writer.add_scalar("eval/episode_reward", episodeReward, action_index)
    writer.add_scalar("eval/episode_length", stepCount, action_index)

    # Use the evaluation metric mentioned in the DQN paper
    with torch.no_grad():
      meanQValue = policy_net(stateSamples).max(1).values.mean()
    writer.add_scalar("eval/mean_max_q_value", meanQValue, action_index)

    return finalPathLength

  def updateTarget():
    print(f'Updating target network (episode #{episode_index}, action #{action_index})')
    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = target_net.state_dict()
    policy_net_state_dict = policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
    target_net.load_state_dict(target_net_state_dict)

  trainEpisodeRewardRunningAverage = common.RunningAverage(RUNNING_AVERAGE_LENGTH)
  trainEpisodeLengthRunningAverage = common.RunningAverage(RUNNING_AVERAGE_LENGTH)
  fpsRunningAverage = common.RunningAverage(RUNNING_AVERAGE_LENGTH)

  # Initialize the environment and get its state
  state, info = env.reset()
  stateTensor = common.observationToTensor(env, state, device)
  episodeReward = 0.0
  episodeStepIndex = 0
  episode_index = 0
  needToEval = False
  needToRender = False
  bestPathLength = 0
  frames = []
  for action_index in range(TOTAL_ACTION_COUNT):
    # Start timing of action step
    actionStartTime = time.perf_counter_ns()

    # Calculate and log current epsilon
    eps_threshold = calculateExplorationEpsilon(action_index)
    writer.add_scalar("Epsilon", eps_threshold, action_index)

    actionTensor = common.select_action(env, stateTensor, policy_net, device, eps_threshold)
    observation, reward, terminated, truncated, _ = env.step(actionTensor.item())
    done = terminated or truncated
    episodeReward += reward

    if terminated:
      nextStateTensor = None
    else:
      nextStateTensor = common.observationToTensor(env, observation, device)

    # Store the transition in memory
    rewardTensor = torch.tensor([reward], device=device).unsqueeze(0)
    if USE_PRIORITIZED_EXPERIENCE_REPLAY:
      memory.push(Transition(stateTensor, actionTensor, nextStateTensor, rewardTensor), float('inf'))
    else:
      memory.push(Transition(stateTensor, actionTensor, nextStateTensor, rewardTensor))

    # Move to the next state
    stateTensor = nextStateTensor

    if (action_index+1) % TRAIN_FREQUENCY == 0:
      # Perform one step of the optimization (on the policy network)
      optimize_model(action_index, TOTAL_ACTION_COUNT)

    if (action_index+1) % TARGET_UPDATE_INTERVAL == 0:
      updateTarget()

    if (action_index+1) % EVAL_FREQUENCY == 0:
      needToEval = True

    if (action_index+1) % RENDER_FREQUENCY == 0:
      needToRender = True

    if done:
      trainEpisodeLengthRunningAverage.add(episodeStepIndex)
      trainEpisodeRewardRunningAverage.add(episodeReward)
      writer.add_scalar("train/episode_reward", trainEpisodeRewardRunningAverage.average(), action_index)
      writer.add_scalar("train/episode_length", trainEpisodeLengthRunningAverage.average(), action_index)
      if needToRender:
        # frameImageOutputDir = 'frames/'
        # visualizer.makeFrame(env, policy_net, action_index, device, frameImageOutputDir)
        visualizer.makeFrame(env, policy_net, action_index, device)
        needToRender = False
      if needToEval:
        pathLength = evalModel(env, policy_net, stateSamples, action_index, writer, device)
        needToEval = False
        if pathLength > bestPathLength:
          # Each time the model does better, save it.
          policy_net.save(f'best_{pathLength}.pt')
          bestPathLength = pathLength
      if (episode_index) % 100 == 0:
        print(f'Episode {episode_index} complete')
      episode_index += 1
      episodeReward = 0.0
      episodeStepIndex = 0
      state, info = env.reset()
      stateTensor = common.observationToTensor(env, state, device)
    else:
      episodeStepIndex += 1

    actionEndTime = time.perf_counter_ns()
    fpsRunningAverage.add(1.0e9 / (actionEndTime-actionStartTime))
    writer.add_scalar("fps", fpsRunningAverage.average(), action_index)

  policy_net.save('policy_net_script.pt')

  outputVideoPath = 'output_video.mp4'
  visualizer.saveVideo(outputVideoPath)
