import pathlib
from typing import Dict

data_dir = pathlib.Path('/workspace/processed_data')
output_dir = pathlib.Path('/workspace/output')

data_count: Dict[str, int] = {'user': 35, 'segment': 20}
state_class_num: int = 3