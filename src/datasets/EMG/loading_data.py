import pathlib
from typing import Optional

import numpy
import pandas
from torch.utils import data as tor_data

from .EMG import EMG
from configs import Configuration as conf

def split_dataset(dataset: tor_data.Dataset, split_ratio: float = 0.1):
    """
    Randomly split the dataset into two datasets.
    
    :param dataset: The dataset to be split.
    :param split_ratio: What percentage of data is kept for the first dataset.
    :return: Two datasets.
    """
    
    # calculate the size of each subset
    dataset_size = len(dataset)
    dataset1_size = int(numpy.floor(split_ratio * dataset_size))
    dataset2_size = dataset_size - dataset1_size
    
    dataset1, dataset2 = tor_data.random_split(dataset, [dataset1_size, dataset2_size])
    
    return dataset1, dataset2

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
    
    train_dataset, other_datasets = split_dataset(datasets, 0.8)
    val_dataset, test_dataset = split_dataset(other_datasets, 0.5)
    
    def _worker_init_fn(worker_id: int):
        """
        Set numpy seed for the DataLoader worker.
        """
        numpy.random.seed(tor_data.get_worker_info().seed % 1000000000 + worker_id)
    
    # data_loader = tor_data.DataLoader(
    #         datasets, batch_size = 16,
    #         shuffle = True, drop_last = False,
    #         num_workers = 1, worker_init_fn = _worker_init_fn, pin_memory = False)
    train_loader = tor_data.DataLoader(
            train_dataset, batch_size = conf.batch_size,
            shuffle = True, drop_last = False,
            num_workers = 1, worker_init_fn = _worker_init_fn, pin_memory = False)
    val_loader = tor_data.DataLoader(
            val_dataset, batch_size = conf.batch_size,
            shuffle = True, drop_last = False,
            num_workers = 1, worker_init_fn = _worker_init_fn, pin_memory = False)
    test_loader = tor_data.DataLoader(
            test_dataset, batch_size = conf.batch_size,
            shuffle = True, drop_last = False,
            num_workers = 1, worker_init_fn = _worker_init_fn, pin_memory = False)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = loading_data(
            data_dir = '/workspace/processed_data',   
            class_num = 3,
            additional_info = True)
    batch = next(iter(train_loader))
    for tensor in batch:
        print(tensor.shape, tensor.dtype)
    for i, data in enumerate(train_loader, 0):
        print(i)
        print(data[0].shape)
    print(len(train_loader))
    