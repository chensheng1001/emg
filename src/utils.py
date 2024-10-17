import numpy as np
import os
import math
import time
import random
import shutil

import torch
from torch import nn


import torchvision.utils as vutils
import torchvision.transforms as standard_transforms

import pdb


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def logger(exp_path, exp_name, work_dir, resume=False):

    from tensorboardX import SummaryWriter
    
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)
    writer = SummaryWriter(exp_path+ '/' + exp_name)
    # log_file = exp_path + '/' + exp_name + '/' + exp_name + '.txt'
    
    # cfg_file = open('./config.py',"r")  
    # cfg_lines = cfg_file.readlines()
    
    # with open(log_file, 'a') as f:
    #     f.write(''.join(cfg_lines) + '\n\n\n\n')

    # if not resume:
    #     copy_cur_env(work_dir, exp_path+ '/' + exp_name + '/code', exception)


    return writer

# def copy_cur_env(work_dir, dst_dir, exception):

#     if not os.path.exists(dst_dir):
#         os.mkdir(dst_dir)

#     for filename in os.listdir(work_dir):

#         file = os.path.join(work_dir,filename)
#         dst_file = os.path.join(dst_dir,filename)


#         if os.path.isdir(file) and exception not in filename:
#             shutil.copytree(file, dst_file)
#         elif os.path.isfile(file):
#             shutil.copyfile(file,dst_file)
