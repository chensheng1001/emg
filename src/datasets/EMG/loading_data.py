import pathlib
from typing import Optional

import numpy
import pandas
from torch.utils import data as tor_data

from EMG import EMG

def loading_data(
        data_dir: pathlib.Path,
        class_num: int,
        key: Optional[str] = 'data',
        additional_info: Optional[bool] = False):
    
    hdf_path = data_dir + '/data_list.hf'
    data: pandas.DataFrame = pandas.read_hdf(hdf_path, key = key)
    if not isinstance(data, pandas.DataFrame):
        message = "The data stored in data_list.hf is a {} instead of pandas.DataFrame".format(type(data))
        raise TypeError(message)  
    datasets = EMG(
            data_dir = data_dir,
            data = data,   
            class_num = class_num,
            additional_info = additional_info)
    
    def _worker_init_fn(worker_id: int):
        """
        Set numpy seed for the DataLoader worker.
        """
        numpy.random.seed(tor_data.get_worker_info().seed % 1000000000 + worker_id)
    
    data_loader = tor_data.DataLoader(
            datasets, batch_size = 16,
            shuffle = True, drop_last = False,
            num_workers = 1, worker_init_fn = _worker_init_fn, pin_memory = False)

    return data_loader

if __name__ == '__main__':
    data_loader = loading_data(
            data_dir = '/workspace/processed_data',   
            class_num = 3,
            additional_info = True)
    batch = next(iter(data_loader))
    for tensor in batch:
        print(tensor.shape, tensor.dtype)
    