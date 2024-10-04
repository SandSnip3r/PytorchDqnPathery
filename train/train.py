
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

def main():
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
  PRIORITIZED_EXPERIENCE_REPLAY_BETA = 0.8
  USE_PRIORITIZED_EXPERIENCE_REPLAY = False

  policy_net = torch.jit.script(common.convFromEnv(env).to(device))
  target_net = torch.jit.script(common.convFromEnv(env).to(device))
  print(f'Policy net: {policy_net}')
  target_net.load_state_dict(policy_net.state_dict())
  target_net.eval()

  # optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
  optimizer = GrokFastAdamW(policy_net.parameters(), lr=LEARNING_RATE)
  if USE_PRIORITIZED_EXPERIENCE_REPLAY:
    # memory = PrioritizedExperienceReplay(MEMORY_CAPACITY, BATCH_SIZE)
    memory = prioritized_buffer.PrioritizedExperienceReplayBufferObject(MEMORY_CAPACITY, BATCH_SIZE, PRIORITIZED_EXPERIENCE_REPLAY_ALPHA)
  else:
    memory = ReplayMemory(MEMORY_CAPACITY)

  stateSamples = getStateSamples(env, STATE_SAMPLE_COUNT, device)

  writer = SummaryWriter()

  def calculateExplorationEpsilon(action_index):
    progress = action_index / TOTAL_ACTION_COUNT
    return EXPLORATION_INITIAL_EPS + (EXPLORATION_FINAL_EPS-EXPLORATION_INITIAL_EPS) * (min(progress, EXPLORATION_FRACTION) / EXPLORATION_FRACTION)

  def optimize_model():
    if len(memory) < max(BATCH_SIZE, STEP_BEFORE_TRAINING):
      return
    # Get a list of Transitions
    if USE_PRIORITIZED_EXPERIENCE_REPLAY:
      transitionSamples = memory.sample(PRIORITIZED_EXPERIENCE_REPLAY_BETA)
      transitions = [x.item for x in transitionSamples]
      indices = [x.dataIndex for x in transitionSamples]
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
        if len(indices) != len(errors):
          raise ValueError(f'Expecting number of indices ({len(indices)}) and errors ({len(errors)}) to be the same')
        for i, memoryIndex in enumerate(indices):
          memory.updatePriority(memoryIndex, errors[i])
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

  def makeFrame(env, policy_net, action_index, device):
    policy_net.eval()
    startColor = (1.0, 0.0, 1.0)
    endColor = (0.0, 0.0, 0.0)

    obstacle_image_path = 'pathery_images/OverlayTileFaceted50b.png'
    obstacle_img = Image.open(obstacle_image_path).resize((50, 50))
    obstacle_img_data = np.asarray(obstacle_img)

    start_image_path = 'pathery_images/OverlayStart50b.png'
    start_img = Image.open(start_image_path).resize((50, 50))
    start_img_data = np.asarray(start_img)

    goal_image_path = 'pathery_images/OverlayFinish50c.png'
    goal_img = Image.open(goal_image_path).resize((50, 50))
    goal_img_data = np.asarray(goal_img)

    ice_image_path = 'pathery_images/PathableOnly1.png'
    ice_img = Image.open(ice_image_path).resize((50, 50))
    ice_img_data = np.asarray(ice_img)

    checkpointImageDatas = []
    # Load checkpoints 'A'-'N'
    for checkpointCharInt in range(ord('A'), ord('N')+1):
      checkpointImagePath = f'pathery_images/Waypoints_{chr(checkpointCharInt)}.png'
      checkpointImg = Image.open(checkpointImagePath).resize((50, 50))
      checkpointImageDatas.append(np.asarray(checkpointImg))

    def getStepCount():
      observation, _ = env.reset()
      observationTensor = common.observationToTensor(env, observation, device)
      stepCount = 0
      while True:
        actionTensor = common.select_action(env, observationTensor, policy_net, device, eps_threshold=None, deterministic=True)
        observation, _, terminated, truncated, _ = env.step(actionTensor.item())
        observationTensor = common.observationToTensor(env, observation, device)
        stepCount += 1
        if terminated or truncated:
          return stepCount

    totalStepCount = getStepCount()

    observation, _ = env.reset()

    outputDir = 'frames/'
    boardObservation = observation[PatheryEnv.OBSERVATION_BOARD_STR]
    # Shape of booard observation is (cellTypeCount, height, width)
    gridSize = boardObservation.shape[1:]
    fig, ax = plt.subplots()
    ax.set_xlim(0, gridSize[1])
    ax.set_ylim(0, gridSize[0])
    ax.set_xticks(np.arange(0, gridSize[1], 1))
    ax.set_yticks(np.arange(0, gridSize[0], 1))
    ax.grid(True)

    # ==================================================================================
    # =================================== Draw data ====================================
    # ==================================================================================

    # Draw the coloring based on the output of the policy net
    # Step through the environment while doing this.
    done = False
    stepIndex = 0
    while not done:
      observationTensor = common.observationToTensor(env, observation, device)
      with torch.no_grad():
        def minMaxNormalize(input, newMin, newMax):
          # Min-max normalization
          inputMin = torch.min(input)
          inputMax = torch.max(input)

          # Avoid division by zero if all elements are the same
          if inputMax > inputMin:
            # Scale to [newMin, newMax]
            inputScaled = newMin + (newMax - newMin) * (input - inputMin) / (inputMax - inputMin)
          else:
            inputScaled = torch.full_like(x, newMin)  # If all elements are the same, set to newMin

          return inputScaled

        netOutput = policy_net(observationTensor).cpu().squeeze()
        # ------------ Topk -----------
        values, indices = torch.topk(netOutput, totalStepCount-stepIndex, sorted=True)
        indices2d = torch.unravel_index(indices, (gridSize[0], gridSize[1]))
        topItems = torch.stack(indices2d, dim=1)
        # ---------- Softmax ----------
        # netOutput = minMaxNormalize(netOutput, 0.0, 1.0)
        # softmaxResult = netOutput.softmax(0).reshape((gridSize[0], gridSize[1]))
        # -----------------------------
      
      def drawPatchAt(color, row, col):
        rect = Rectangle((col, gridSize[0]-row-1), 1, 1, facecolor=color, zorder=1)
        ax.add_patch(rect)

      def colorScale(value, start_color, end_color):
        """
        Scales between two colors based on the input value in the range [0.0, 1.0].

        Args:
            value (float): A number in the range [0.0, 1.0] representing the position in the scale.
            start_color (tuple): The RGB tuple of the starting color.
            end_color (tuple): The RGB tuple of the ending color.

        Returns:
            tuple: An (R, G, B) tuple representing the interpolated color.
        """
        # Clamp the value between 0 and 1
        value = max(0.0, min(1.0, value))

        # Interpolate between start_color and end_color
        red = (1 - value) * start_color[0] + value * end_color[0]
        green = (1 - value) * start_color[1] + value * end_color[1]
        blue = (1 - value) * start_color[2] + value * end_color[2]

        return (red, green, blue)
      
      # ---------------------------- All grid ----------------------------

      # for row in range(gridSize[0]):
      #   for col in range(gridSize[1]):
      #     if totalStepCount == 1:
      #       color = colorScale(0.0, startColor, endColor)
      #     else:
      #       color = colorScale(stepIndex/(totalStepCount-1), startColor, endColor)
      #     greenColor = (*color, float(softmaxResult[row][col]))
      #     drawPatchAt(greenColor, row, col)

      # ------------------------------ Topk ------------------------------

      for index, (row, col) in enumerate(topItems):
        if index == 0:
          # First choice is a solid color
          finalColor = (*startColor, 1.0)
        else:
          color = colorScale(index/(len(topItems)-1), startColor, endColor)
          finalColor = (*color, 0.1)
        drawPatchAt(finalColor, row, col)

      # ---------------------------- Just max ----------------------------

      # index2d = (softmaxResult==torch.max(softmaxResult)).nonzero()[0]
      # if totalStepCount == 1:
      #   color = colorScale(0.0, startColor, endColor)
      # else:
      #   color = colorScale(stepIndex/(totalStepCount-1), startColor, endColor)
      # greenColor = (*color, 1.0)
      # drawPatchAt(greenColor, index2d[0], index2d[1])

      # ------------------------------------------------------------------

      # Step to next state
      actionTensor = common.select_action(env, observationTensor, policy_net, device, eps_threshold=None, deterministic=True)
      observation, _, terminated, truncated, _ = env.step(actionTensor.item())
      done = terminated or truncated
      stepIndex += 1

    # ==================================================================================
    # =================================== Draw board ===================================
    # ==================================================================================

    def drawImageAt(img, row, col):
      ax.imshow(img, extent=[col, col+1, gridSize[0] - row - 1, gridSize[0] - row], zorder=2)

    # Draw the blocks of the board
    # TODO(maybe): Walls are not drawn (rocks are)

    rockLayer = boardObservation[CellType.ROCK.value]
    for row in range(len(rockLayer)):
      for col in range(len(rockLayer[row])):
        if rockLayer[row][col]:
          drawImageAt(obstacle_img_data, row, col)

    startLayer = boardObservation[CellType.START.value]
    for row in range(len(startLayer)):
      for col in range(len(startLayer[row])):
        if startLayer[row][col]:
          drawImageAt(start_img_data, row, col)

    goalLayer = boardObservation[CellType.GOAL.value]
    for row in range(len(goalLayer)):
      for col in range(len(goalLayer[row])):
        if goalLayer[row][col]:
          drawImageAt(goal_img_data, row, col)

    iceLayer = boardObservation[CellType.ICE.value]
    for row in range(len(iceLayer)):
      for col in range(len(iceLayer[row])):
        if iceLayer[row][col]:
          drawImageAt(ice_img_data, row, col)

    firstCheckpointLayerIndex = CellType.ICE.value+1
    for checkpointLayerIndex in range(firstCheckpointLayerIndex, len(boardObservation)):
      checkpointLayer = boardObservation[checkpointLayerIndex]
      for row in range(len(checkpointLayer)):
        for col in range(len(checkpointLayer[row])):
          if checkpointLayer[row][col]:
            drawImageAt(checkpointImageDatas[checkpointLayerIndex-firstCheckpointLayerIndex], row, col)

    # ==================================================================================
    # =================================== Draw text ====================================
    # ==================================================================================

    text_x = gridSize[1] / 2
    text_y = gridSize[0] + 0.5
    plt.text(x=text_x, y=text_y, s=f'Total Move Count: {totalStepCount}; action #{action_index}', 
         fontsize=12, ha='center', va='center')

    # Create a secondary axis for the text and gradient (position it outside the grid)
    gradient_ax = fig.add_axes([0.1, 0.85, 0.8, 0.1], frameon=False)  # Custom position
    gradient_ax.set_xticks([])
    gradient_ax.set_yticks([])
    gradient_ax.set_xlim(0, 1)
    gradient_ax.set_ylim(0, 1)

    # ==================================================================================
    # ================================== Draw Legend ===================================
    # ==================================================================================

    # Add text to the secondary axis
    gradient_ax.text(0.0, 0.5, s="Gradient Legend:", fontsize=12, ha='left', va='center')

    # Parameters for the gradient blocks
    N = totalStepCount  # Number of blocks
    total_width = 0.6  # Total width (fraction of the secondary axis)
    block_width = total_width / N
    block_height = 0.4  # Height of each block (fraction of the secondary axis)
    start_x = 0.35  # Start of the gradient, in axis coordinates
    start_y = 0.3  # Vertical position for the blocks

    # Draw gradient rectangles on the secondary axis
    for i in range(N):
      # Determine the color from green-to-red scale
      if N == 1:
        value = 0.0
      else:
        value = i / (N - 1)  # Linearly interpolate between 0 and 1
      color = colorScale(value, startColor, endColor)

      # Create the rectangle
      rect = Rectangle((start_x + i * block_width, start_y), block_width, block_height, 
                              # linewidth=1, edgecolor='black',
                              facecolor=color)

      # Add rectangle to the secondary axis
      gradient_ax.add_patch(rect)

    # Save figure as image
    # filename = os.path.join(outputDir, f'{action_index}.png')
    # plt.savefig(filename)

    # Render to buffer instead of saving to disk
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Read the image from buffer and convert to numpy array
    img = Image.open(buf)
    return np.array(img)

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
      optimize_model()

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
        frames.append(makeFrame(env, policy_net, action_index, device))
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

  # Create a video writer using imageio
  output_video_path = 'output_video.mp4'
  with imageio.get_writer(output_video_path, fps=RENDER_FPS, format='mp4') as writer:
    for frame in frames:
      # Write the frame to the video (converting to uint8 for imageio)
      writer.append_data(frame.astype(np.uint8))
  print(f"Video saved at {output_video_path}")

if __name__ == "__main__":
  main()
