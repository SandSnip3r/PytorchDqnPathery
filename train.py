#!/usr/bin/env python

import re
import sys
import unicodedata

from train import train

def slugify(value, allow_unicode=False):
  value = str(value)
  if allow_unicode:
    value = unicodedata.normalize('NFKC', value)
  else:
    value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
  value = re.sub(r'[^\w\s-]', '', value.lower())
  return re.sub(r'[-\s]+', '-', value).strip('-_')

if __name__ == "__main__":
  if len(sys.argv) > 1:
    filename = slugify(sys.argv[1])
    print(f'Using experiment name "{filename}"')
    train.main(filename)
  else:
    print(f'No experiment name given')
    train.main()