import copy
import dataclasses
import pathlib
from typing import Dict, List, Tuple, Union

from configs2 import state_class_num, data_count, data_dir, output_dir


@dataclasses.dataclass
class Configuration:
    """All configurations."""
    # paths
    data_dir: pathlib.Path = copy.deepcopy(data_dir)
    output_dir: pathlib.Path = copy.deepcopy(output_dir)
   
    data_count: Dict[str, int] = dataclasses.field(default_factory = data_count.copy)
    class_num: int = state_class_num

    seed: int = 3035
    gpu_id = [0]

    batch_size = 16

    net = 'CNN'
    learning_rate = 1e-5
    learning_rate_decay = 0.99
    num_epoch_lr_decay = 1
    start_lr_decay = -1

    is_load = False
    load_model_path = ""

    exp_path = '/workspace/exp'
    exp_name = 'cnn'

    max_epoch = 100
    
    val_freq = 10
    val_every_start = 50

    print_freq = 10


default_configs = Configuration()
  