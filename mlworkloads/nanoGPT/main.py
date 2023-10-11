import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from profiler  import AverageMeter, ProgressMeter
# ----------------------------------------------------
def main():
    print()

    
if __name__ == "__main__":
    main()