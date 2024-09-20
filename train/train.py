
import numpy as np
import copy
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler

from collections import namedtuple, deque
from common import common
from torch.utils.tensorboard import SummaryWriter
from pathery_env.envs.pathery import PatheryEnv

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
  result = []
  for i in range(0, stateCount, env.num_envs):
    action = actionSpace.sample()
    observation, _, terminated, truncated, info = env.step(action)
    done = np.logical_or(terminated, truncated)
    for index, value in enumerate(done):
      if value:
        # Episode reset, get observation from final
        result.append(common.observationToTensor(env, info['final_observation'][index], device))
      else:
        # Take normal observation
        result.append(common.observationToTensor(env, { PatheryEnv.OBSERVATION_BOARD_STR: observation[PatheryEnv.OBSERVATION_BOARD_STR][index] }, device))
  res = torch.cat(result)
  return res

def main():
  device = common.getDevice()
  print(f'Using device {device}')

  env = common.getEnv()

  # Seed the random number generation
  INITIAL_SEED = 123
  set_seed(INITIAL_SEED, env)

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
  GAMMA = 0.99
  EXPLORATION_INITIAL_EPS = 1.0
  EXPLORATION_FINAL_EPS = 0.05
  EXPLORATION_FRACTION = 0.1
  TAU = 0.95 # 0.005
  LEARNING_RATE = 1e-4
  TARGET_UPDATE_INTERVAL = 10000
  TRAIN_FREQUENCY = 4
  RUNNING_AVERAGE_LENGTH = 128
  EVAL_FREQUENCY = 1000
  STATE_SAMPLE_COUNT = 512
  DOUBLE_DQN = True
  TOTAL_ACTION_COUNT = 20_000_000

  policy_net = torch.jit.script(common.convFromEnv(env).to(device))
  target_net = torch.jit.script(common.convFromEnv(env).to(device))
  print(f'Policy net: {policy_net}')
  target_net.load_state_dict(policy_net.state_dict())

  optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
  memory = ReplayMemory(100000)

  stateSamples = getStateSamples(env, STATE_SAMPLE_COUNT, device)
  # Copy the env once, so we can reuse it for eval
  # evalEnv = copy.deepcopy(env)

  writer = SummaryWriter()

  def calculateExplorationEpsilon(actionIndex):
    progress = actionIndex / TOTAL_ACTION_COUNT
    return EXPLORATION_INITIAL_EPS + (EXPLORATION_FINAL_EPS-EXPLORATION_INITIAL_EPS) * (min(progress, EXPLORATION_FRACTION) / EXPLORATION_FRACTION)

  def optimize_model():
    if len(memory) < BATCH_SIZE:
      return
    # Get a list of Transitions
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays (specifically, they're tuples).
    batch = Transition(*zip(*transitions))

    if all(s is None for s in batch.next_state):
      # We require at least one next action for any model training
      return

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: [s is not None],
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

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
        next_state_actions = policy_net(non_final_next_states).max(1).indices
        # Action evaluation using the target network (target_net)
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_state_actions.unsqueeze(1)).squeeze(1)
      else:
        # Traditional DQN
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

  def evalModel(env, policy_net, stateSamples, actionIndex, writer, device):
    # Deterministically use the model to play one episode. Log the length & reward.
    observations, info = env.reset()
    observationTensors = common.vecObservationToTensors(env, observations, device)
    episodeRewards = np.zeros(env.num_envs)
    stepCount = 0
    while True:
      vectorizedObervationsTensor = torch.cat([obsTensor for obsTensor in observationTensors])
      actionTensors = common.vecSelectActions(env, vectorizedObervationsTensor, policy_net, device, epsilonThreshold=None, deterministic=True)
      observations, rewards, terminated, truncated, info = env.step(actionTensors)
      observationTensors = common.vecObservationToTensors(env, observations, device)
      episodeRewards += rewards
      stepCount += 1
      if any([any(terminated), any(truncated)]):
        break
    
    rewardSet = {r for r in episodeRewards}
    if len(rewardSet) > 1:
      raise ValueError(f'Expecting only one reward, got {episodeRewards}')
    
    episodeReward = episodeRewards.mean()

    writer.add_scalar("eval/episode_reward", episodeReward, actionIndex)
    writer.add_scalar("eval/episode_length", stepCount, actionIndex)

    # Use the evaluation metric mentioned in the DQN paper
    with torch.no_grad():
      meanQValue = policy_net(stateSamples).max(1).values.mean()
    writer.add_scalar("eval/mean_max_q_value", meanQValue, actionIndex)

    return episodeReward

  def updateTarget(actionIndex, episodeCount):
    print(f'Updating target network ({episodeCount} episode(s) completed, action #{actionIndex})')
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
  initialObservations, _ = env.reset()
  currentObservationTensors = common.vecObservationToTensors(env, initialObservations, device)

  # Track some episode data
  lastOptimizationStep = -1
  lastUpdateTargetStep = -1
  lastEvalStep = -1
  episodeRewards = np.zeros(env.num_envs)
  episodeStepIndices = np.zeros(env.num_envs, dtype=np.int32)
  episodeCounts = np.zeros(env.num_envs, dtype=np.int32)
  bestEvalReward = 0

  # Main training loop
  for actionIndex in range(0, TOTAL_ACTION_COUNT, env.num_envs):
    # Start timing of action step
    actionStartTime = time.perf_counter_ns()

    # Calculate and log current epsilon
    epsilonThreshold = calculateExplorationEpsilon(actionIndex)
    writer.add_scalar("Epsilon", epsilonThreshold, actionIndex)

    vectorizedObervationsTensor = torch.cat([obsTensor for obsTensor in currentObservationTensors])
    actionTensors = common.vecSelectActions(env, vectorizedObervationsTensor, policy_net, device, epsilonThreshold)
    observations, rewards, vecTerminated, vecTruncated, infos = env.step(actionTensors)

    vecDone = np.logical_or(vecTerminated, vecTruncated)
    episodeCounts[vecDone] += np.ones(episodeCounts.shape, dtype=np.int32)[vecDone]

    episodeRewards += rewards
    episodeStepIndices += np.ones(episodeStepIndices.shape, dtype=np.int32)

    # Since vectorized environments auto-reset, in order to create (S,A,R,S) tuples, we might need to look into the `info` to get the "final" observation of an auto-reset env.
    finalObservationMaskKey = '_final_observation'
    finalObservationKey = 'final_observation'
    if finalObservationMaskKey in infos:
      # There were some envs which were auto-reset
      nextObservationTensors = []
      for index, wasFinal in enumerate(infos[finalObservationMaskKey]):
        if wasFinal:
          nextObservationTensors.append(common.observationToTensor(env, infos[finalObservationKey][index], device))
        else:
          # Since vectorized environments' observations are a dict of arrays: un-dict, index into the array, then re-dict the single observation
          boards = observations[PatheryEnv.OBSERVATION_BOARD_STR]
          dictedBoard = { PatheryEnv.OBSERVATION_BOARD_STR: boards[index] }
          nextObservationTensors.append(common.observationToTensor(env, dictedBoard, device))
    else:
      # No env reset, take all observations directly from `step` output
      nextObservationTensors = common.vecObservationToTensors(env, observations, device)

    # Turn the rewards into tensors
    rewardTensors = []
    for i in range(len(rewards)):
      rewardTensors.append(torch.tensor(rewards[i:i+1], device=device))

    # Make sure that all lists have the same length
    lengths = {len(lst) for lst in [currentObservationTensors, actionTensors, nextObservationTensors, rewardTensors]}
    if len(lengths) != 1:
      raise ValueError(f'Mismatched lengths: {len(currentObservationTensors)}, {len(actionTensors)}, {len(nextObservationTensors)}, {len(rewardTensors)}')

    # Store the transitions in memory
    for i in range(len(currentObservationTensors)):
      memory.push(currentObservationTensors[i], actionTensors[i], nextObservationTensors[i], rewardTensors[i])

    # Move to the next state
    currentObservationTensors = nextObservationTensors

    for index, value in enumerate(vecDone):
      if not value:
        # This env's episode did not complete
        continue
      # This env's episode completed; write stats
      trainEpisodeRewardRunningAverage.add(episodeRewards[index])
      trainEpisodeLengthRunningAverage.add(episodeStepIndices[index])
      writer.add_scalar("train/episode_reward", trainEpisodeRewardRunningAverage.average(), actionIndex)
      writer.add_scalar("train/episode_length", trainEpisodeLengthRunningAverage.average(), actionIndex)
      # Reset values for this env
      episodeRewards[index] = 0
      episodeStepIndices[index] = 0

    if actionIndex-lastOptimizationStep >= TRAIN_FREQUENCY:
      # Perform one step of the optimization (on the policy network)
      optimize_model()
      lastOptimizationStep = actionIndex

    if actionIndex-lastUpdateTargetStep >= TARGET_UPDATE_INTERVAL:
      updateTarget(actionIndex, episodeCounts.sum())
      lastUpdateTargetStep = actionIndex

    # if actionIndex-lastEvalStep >= EVAL_FREQUENCY:
    #   # Evaluate
    #   evalReward = evalModel(evalEnv, policy_net, stateSamples, actionIndex, writer, device)
    #   if evalReward > bestEvalReward:
    #     # Each time the model does better, save it.
    #     print(f'New best with {evalReward}')
    #     # policy_net.save(f'best_{evalReward}.pt')
    #     bestEvalReward = evalReward

    actionEndTime = time.perf_counter_ns()
    fpsRunningAverage.add((1.0e9 / (actionEndTime-actionStartTime)) * env.num_envs)
    writer.add_scalar("fps", fpsRunningAverage.average(), actionIndex)

  policy_net.save('policy_net_script.pt')

if __name__ == "__main__":
  main()
