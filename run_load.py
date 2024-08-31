#!/usr/bin/env python

import argparse

from load import load

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Load a pathery model.")
  parser.add_argument('--model', type=str, help='The model file with extension .pt')

  args = parser.parse_args()

  if args.model is None:
    print(f'No model given')
    quit()
  load.main(args.model)