
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from common import common
from io import BytesIO
from matplotlib.patches import Rectangle
from pathery_env.envs.pathery import PatheryEnv
from pathery_env.envs.pathery import CellType
from PIL import Image

class Visualizer:
  def __init__(self, showOriginalColors: bool=False, videoFps: int=30):
    self.patheryImageDirectory = '../pathery_images'
    self.frames = []
    self.showOriginalColors = showOriginalColors
    self.videoFps = videoFps

    self.parsePatheryImageData()

  def parsePatheryImageData(self):
    obstacleImagePath = 'pathery_images/OverlayTileFaceted50b.png'
    startImagePath = 'pathery_images/OverlayStart50b.png'
    goalImagePath = 'pathery_images/OverlayFinish50c.png'
    iceImagePath = 'pathery_images/PathableOnly1.png'

    obstacleImg = Image.open(obstacleImagePath).resize((50, 50))
    startImg = Image.open(startImagePath).resize((50, 50))
    goalImg = Image.open(goalImagePath).resize((50, 50))
    iceImg = Image.open(iceImagePath).resize((50, 50))

    self.cellImageDict = {}
    self.cellImageDict[CellType.ROCK.value] = np.asarray(obstacleImg)
    self.cellImageDict[CellType.START.value] = np.asarray(startImg)
    self.cellImageDict[CellType.GOAL.value] = np.asarray(goalImg)
    self.cellImageDict[CellType.ICE.value] = np.asarray(iceImg)

    self.cellImageBackgroundColor = {}
    self.cellImageBackgroundColor[CellType.ROCK.value] = '#B85555'
    self.cellImageBackgroundColor[CellType.START.value] = '#FBFEFB'
    self.cellImageBackgroundColor[CellType.GOAL.value] = '#666666'
    self.cellImageBackgroundColor[CellType.ICE.value] = '#44FFFF'

    self.checkpointImageDatas = []
    # Load checkpoints 'A'-'N'
    for checkpointCharInt in range(ord('A'), ord('N')+1):
      checkpointImagePath = f'pathery_images/Waypoints_{chr(checkpointCharInt)}.png'
      checkpointImg = Image.open(checkpointImagePath).resize((50, 50))
      self.checkpointImageDatas.append(np.asarray(checkpointImg))

    # Checkpoint colors
    self.checkpointColors = []
    self.checkpointColors.append('#F777FF')
    self.checkpointColors.append('#FFFF11')
    self.checkpointColors.append('#FF4466')
    self.checkpointColors.append('#FF9911')
    self.checkpointColors.append('#00FFFF')
    self.checkpointColors.append('#A12EC4')
    self.checkpointColors.append('#46C0A0')
    self.checkpointColors.append('#33FF33')
    self.checkpointColors.append('#F032E6')
    self.checkpointColors.append('#D2F53C')
    self.checkpointColors.append('#FABEBE')
    self.checkpointColors.append('#9090F4')
    self.checkpointColors.append('#E6BEFF')
    self.checkpointColors.append('#AA6E28')

    teleportInWhite = 'pathery_images/TeleportInW.png'
    teleportOutWhite = 'pathery_images/TeleportOutW.png'
    teleportInBlack = 'pathery_images/TeleportIn.png'
    teleportOutBlack = 'pathery_images/TeleportOut.png'
    teleportInWhiteImg = Image.open(teleportInWhite).resize((50, 50))
    teleportOutWhiteImg = Image.open(teleportOutWhite).resize((50, 50))
    teleportInBlackImg = Image.open(teleportInBlack).resize((50, 50))
    teleportOutBlackImg = Image.open(teleportOutBlack).resize((50, 50))
    self.teleportImageData = []
    self.teleportImageData.append(np.asarray(teleportInWhiteImg))
    self.teleportImageData.append(np.asarray(teleportOutWhiteImg))
    self.teleportImageData.append(np.asarray(teleportInBlackImg))
    self.teleportImageData.append(np.asarray(teleportOutBlackImg))

    self.teleportColors = []
    self.teleportColors.append('#3377AA')
    self.teleportColors.append('#44EE66')
    self.teleportColors.append('#CC5544')
    self.teleportColors.append('#55CCFF')
    self.teleportColors.append('#005533')
    self.teleportColors.append('#FFFFFF')
    self.teleportColors.append('#000000')

  def makeFrame(self, env, policy_net, actionIndex, device, frameImageOutputDir: str = None):
    policy_net.eval()
    startColor = (1.0, 0.0, 1.0)
    endColor = (0.0, 0.0, 0.0)

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
            inputScaled = torch.full_like(input, newMin)  # If all elements are the same, set to newMin

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
        rect = Rectangle((col, gridSize[0]-row-1), 1, 1, facecolor=color, zorder=2)
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

    def drawImageAt(img, row, col, backgroundColor=None):
      # Lower zorder drawn first
      if backgroundColor is not None:
        rect = Rectangle((col, gridSize[0]-row-1), 1, 1, facecolor=backgroundColor, zorder=1)
        ax.add_patch(rect)
      ax.imshow(img, extent=[col, col+1, gridSize[0] - row - 1, gridSize[0] - row], zorder=3)

    # Draw the board
    # Note: Walls are not drawn (rocks are)
    for cellType in [CellType.ROCK.value, CellType.START.value, CellType.GOAL.value, CellType.ICE.value]:
      boardLayer = boardObservation[cellType]
      for row in range(len(boardLayer)):
        for col in range(len(boardLayer[row])):
          if boardLayer[row][col]:
            if self.showOriginalColors:
              drawImageAt(self.cellImageDict[cellType], row, col, self.cellImageBackgroundColor[cellType])
            else:
              drawImageAt(self.cellImageDict[cellType], row, col)

    # Draw any checkpoints
    firstCheckpointLayerIndex = CellType.ICE.value+1
    firstTeleporterIndex = firstCheckpointLayerIndex+env.unwrapped.maxCheckpointCount
    # len(boardObservation)
    for checkpointLayerIndex in range(firstCheckpointLayerIndex, firstTeleporterIndex):
      checkpointIndex = checkpointLayerIndex - firstCheckpointLayerIndex
      checkpointLayer = boardObservation[checkpointLayerIndex]
      for row in range(len(checkpointLayer)):
        for col in range(len(checkpointLayer[row])):
          if checkpointLayer[row][col]:
            # self.checkpointColors
            if self.showOriginalColors:
              drawImageAt(self.checkpointImageDatas[checkpointIndex], row, col, self.checkpointColors[checkpointIndex])
            else:
              drawImageAt(self.checkpointImageDatas[checkpointIndex], row, col)

    # Draw any teleporters
    for teleportLayerIndex in range(firstTeleporterIndex, len(boardObservation)):
      teleportIndex = teleportLayerIndex - firstTeleporterIndex
      teleportLayer = boardObservation[teleportLayerIndex]
      for row in range(len(teleportLayer)):
        for col in range(len(teleportLayer[row])):
          if teleportLayer[row][col]:
            if self.showOriginalColors:
              drawImageAt(self.teleportImageData[teleportIndex%len(self.teleportImageData)], row, col, self.teleportColors[teleportIndex//2])
            else:
              drawImageAt(self.teleportImageData[teleportIndex%len(self.teleportImageData)], row, col)

    # ==================================================================================
    # =================================== Draw text ====================================
    # ==================================================================================

    text_x = gridSize[1] / 2
    text_y = gridSize[0] + 0.5
    plt.text(x=text_x, y=text_y, s=f'Total Move Count: {totalStepCount}; action #{actionIndex}', 
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

    if frameImageOutputDir is not None:
      # Save figure as image
      filename = os.path.join(frameImageOutputDir, f'{actionIndex}.png')
      plt.savefig(filename)

    # Render to buffer instead of saving to disk
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Read the image from buffer and convert to numpy array
    img = Image.open(buf)
    self.frames.append(np.array(img, dtype=np.uint8))
  
  def saveVideo(self, filename):
    # Create a video writer using imageio
    # output_video_path = 'output_video.mp4'
    with imageio.get_writer(filename, fps=self.videoFps, format='mp4') as writer:
      for frame in self.frames:
        # Write the frame to the video (converting to uint8 for imageio)
        writer.append_data(frame)
    print(f"Video saved at {filename}")
