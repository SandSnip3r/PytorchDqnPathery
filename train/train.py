
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.profiler

from collections import namedtuple, deque
from common import common
from torch.utils.tensorboard import SummaryWriter

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
  result = [common.observationToTensor(env, observation, device)]
  for i in range(stateCount-1):
    action = actionSpace.sample()
    observation, _, terminated, truncated, _ = env.step(action)
    result.append(common.observationToTensor(env, observation, device))
    if terminated or truncated:
      # Since we're not usually working with random environments, do not add the starting state again
      env.reset()

  return torch.cat(result)

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
  TRAIN_FREQUENCY = 1
  RUNNING_AVERAGE_LENGTH = 128
  EVAL_FREQUENCY = 1000
  STATE_SAMPLE_COUNT = 512
  DOUBLE_DQN = True
  TOTAL_ACTION_COUNT = 500_000

  policy_net = torch.jit.script(common.convFromEnv(env).to(device))
  target_net = torch.jit.script(common.convFromEnv(env).to(device))
  print(f'Policy net: {policy_net}')
  target_net.load_state_dict(policy_net.state_dict())

  optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
  memory = ReplayMemory(100000)

  stateSamples = getStateSamples(env, STATE_SAMPLE_COUNT, device)

  writer = SummaryWriter()

  def calculateExplorationEpsilon(action_index):
    progress = action_index / TOTAL_ACTION_COUNT
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

  def evalModel(env, policy_net, stateSamples, action_index, writer, device):
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

    writer.add_scalar("eval/episode_reward", episodeReward, action_index)
    writer.add_scalar("eval/episode_length", stepCount, action_index)

    # Use the evaluation metric mentioned in the DQN paper
    with torch.no_grad():
      meanQValue = policy_net(stateSamples).max(1).values.mean()
    writer.add_scalar("eval/mean_max_q_value", meanQValue, action_index)

    return episodeReward

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
  bestEvalReward = 0
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
    memory.push(stateTensor, actionTensor, nextStateTensor, rewardTensor)

    # Move to the next state
    stateTensor = nextStateTensor

    if (action_index+1) % TRAIN_FREQUENCY == 0:
      # Perform one step of the optimization (on the policy network)
      optimize_model()

    if (action_index+1) % TARGET_UPDATE_INTERVAL == 0:
      updateTarget()

    if (action_index+1) % EVAL_FREQUENCY == 0:
      needToEval = True

    if done:
      trainEpisodeLengthRunningAverage.add(episodeStepIndex)
      trainEpisodeRewardRunningAverage.add(episodeReward)
      writer.add_scalar("train/episode_reward", trainEpisodeRewardRunningAverage.average(), action_index)
      writer.add_scalar("train/episode_length", trainEpisodeLengthRunningAverage.average(), action_index)
      if needToEval:
        evalReward = evalModel(env, policy_net, stateSamples, action_index, writer, device)
        needToEval = False
        if evalReward > bestEvalReward:
          # Each time the model does better, save it.
          policy_net.save(f'best_{evalReward}.pt')
          bestEvalReward = evalReward
      if (episode_index+1) % 100 == 0:
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

if __name__ == "__main__":
  main()
