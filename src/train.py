import os
import numpy as np
import torch
import argparse
import time

from configs import default_configs as conf
from datasets.EMG.loading_data import loading_data
from trainer import Trainer

# configuration
seed = conf.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = conf.gpu_id
if len(gpus)==1:
    torch.cuda.set_device(gpus[0])

torch.backends.cudnn.benchmark = True

# Training
pwd = os.path.split(os.path.realpath(__file__))[0]
emg_trainer = Trainer(loading_data, pwd)
emg_trainer.forward()
