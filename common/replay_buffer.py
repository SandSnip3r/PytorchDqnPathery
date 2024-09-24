import io
import numpy as np
from collections import deque

def printToString(*args, **kwargs):
  output = io.StringIO()
  print(*args, file=output, **kwargs)
  contents = output.getvalue()
  output.close()
  return contents

class PrioritizedExperienceReplay():
  '''This is the rank-based variant of Prioritized Experience Replay.'''

  def __init__(self, capacity: int, sampleSize: int, alpha: float=1.0):
    self.memory = deque([], maxlen=capacity)
    self.sampleSize = sampleSize
    self.alpha = alpha
    self.currentSize = 0
    self.exclusiveBucketEnds : list[int] = []

  def recomputeBounds(self):
    # Compute buckets
    self.exclusiveBucketEnds.clear()
    sum = 0.0
    for i in range(len(self.memory)):
      sum += (1.0/(i+1))**self.alpha
    cumulativeSum = 0.0
    currentBucket = 0
    for i in range(len(self.memory)):
      priority = (1.0/(i+1))**self.alpha
      probability = priority/sum
      cumulativeSum += probability
      if cumulativeSum >= (currentBucket+1) / self.sampleSize:
        self.exclusiveBucketEnds.append(i+1)
        currentBucket += 1
        if currentBucket == self.sampleSize-1:
          self.exclusiveBucketEnds.append(len(self.memory))
          break

  def push(self, transition, priority:float):
    """Save a transition"""
    self.memory.append([transition, priority])
    if len(self.memory) >= self.sampleSize and len(self.memory) != self.currentSize:
      # List grew, recompute CDF bounds
      self.recomputeBounds()
      self.currentSize = len(self.memory)

  def sample(self) -> list:
    if len(self.memory) < self.sampleSize:
      raise ValueError(f'Trying to sample {self.sampleSize}, but only have {len(self.memory)} item(s)')
    
    errorsAndIndices = sorted([(transitionAndPriority[1], index) for index, transitionAndPriority in enumerate(self.memory)], reverse=True)
    
    lastEnd = 0
    result: list = []
    for end in self.exclusiveBucketEnds:
      indexInBucket = np.random.randint(lastEnd, end)
      _, index = errorsAndIndices[indexInBucket]
      result.append((self.memory[index][0], index))
      lastEnd = end
    return result
  
  def updatePriorities(self, index:int, newPriority:float):
    # print(f'Updating priority of {index} from {self.memory[index][1]} to {newPriority}')
    self.memory[index][1] = newPriority

  def __len__(self):
    return len(self.memory)
  
  def __str__(self):
    return printToString('PrioritizedExperienceReplay:', *self.memory, sep='\n  ')