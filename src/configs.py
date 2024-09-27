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


default_configs = Configuration()
  