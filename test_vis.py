#!/usr/bin/env python

from common import common
from common import Visualizer

if __name__ == "__main__":
  env = common.getEnv()
  visualizer = Visualizer(showOriginalColors=False, videoFps=30)
  policyNet = common.convFromEnv(env)
  visualizer.makeFrame(env, policyNet, 0, 'cpu', 'test_frames')