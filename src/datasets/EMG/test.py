import pathlib
from typing import List, NoReturn, Optional, Tuple

import numpy
import pandas
import scipy.io
import torch
import pickle5 as pickle
from torch.utils import data as tor_data

from GAIT import GAIT, GramSlicer

def loading_data(
        data_dir: pathlib.Path,
        phase_data_dir: pathlib.Path,
        class_num: int,
        gram_type: str,
        key: Optional[str] = 'data',
        additional_labels: Optional[bool] = False,
        slicing: Optional[bool] = False,
        slice_length: Optional[int] = None,
        slice_stride: Optional[int] = None) -> Tuple[tor_data.Dataset, ...]:
    """
    Get datasets from the data folder.
    
    todo multiple domain category.
    
    :param data_dir: The path to the folder that contains all data.
    :param data_dir: The path to the folder that contains all phase data.
    :param key: The pandas object group identifier of data list that will get loaded. Must be `'data'` or `'random'`.
    :param additional_labels: Should the dataset return additional labels?
    :param class_num: The number of classes.
    :param gram_type: which type of gram is used for features.
    :param rooms_list: A list of list of rooms that each returned dataset will contain.
    :param slicing: Slice the gram or not.
    :param slice_length: Snippet length.
    :param slice_stride: Slicing stride.
    :return: `len(rooms_list)` datasets.
    """
    
    if slicing:
        if slice_length is None or slice_stride is None:
            message = "Gram can not be sliced without slice length and slice stride."
            raise ValueError(message)
        slicer = GramSlicer(slice_length, slice_stride)
    else:
        slicer = None

    # read data list
    data: pandas.DataFrame = pandas.read_hdf(data_dir / 'data_list.hf', key = key)
    # data_dict = {"gram":["stft", "hht", "log_stft", "log_hht", "ampl"],
    #                        "room":[1, 2, 3, 4],
    #                        "user":[1, 2, 3, 4, 5, 6, 7, 8]}
    # data = pandas.DataFrame(pandas.DataFrame.from_dict(data_dict, orient='index').values.T, columns=list(data_dict.keys()))
    if not isinstance(data, pandas.DataFrame):
        message = "The data stored in data_list.hf is a {} instead of pandas.DataFrame".format(type(data))
        raise TypeError(message)
    
    datasets = GAIT(data_dir, data, phase_data_dir, class_num, gram_type, slicer, additional_labels)

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
            data_dir = pathlib.Path('/workspace/data/Gait_WiFi/gram_data_v6/all_gram'),
            phase_data_dir = pathlib.Path('/workspace/data/Gait_WiFi/gram_phase_data/all_gram'),
            class_num = 8,
            gram_type = "log_stft",
            slicing = False,
            slice_length = 400, slice_stride = 200, additional_labels = True)
    batch = next(iter(data_loader))
    for tensor in batch:
        print(tensor.shape, tensor.dtype)