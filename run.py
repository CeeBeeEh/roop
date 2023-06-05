#!/usr/bin/env python3

import torch
from roop import core

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    core.run()
