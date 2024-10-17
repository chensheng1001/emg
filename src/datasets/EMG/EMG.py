import pathlib
from typing import Tuple

import numpy
import pandas
import scipy.io
import torch
from torch.utils import data as tor_data


class EMG(tor_data.Dataset):
    def __init__(self,
                 data_dir: pathlib.Path, data: pandas.DataFrame,
                 class_num: int,
                 additional_info: bool):
        self.data_dir = data_dir

        self.sample_names: numpy.ndarray = data['data_name'].to_numpy()
        self.state_labels: numpy.ndarray = data['class'].to_numpy()
        self.user_labels: numpy.ndarray = data['user'].to_numpy()

        self.gram_name = 'gram_stft'
        self.processed_signals = 'processed_signals'
                
        self.class_num = class_num
        self.additional_info = additional_info
        
    def __len__(self) -> int:
        return len(self.user_labels)
    
    def __getitem__(self, ind) -> Tuple[torch.Tensor, torch.Tensor]:
        # read gram mat
        sample_name = self.sample_names[ind]
        sample_path = self.data_dir + '/' + sample_name
        gram_sample = scipy.io.loadmat(sample_path, appendmat = False, variable_names = [self.gram_name])[self.gram_name]
        signal_sample = scipy.io.loadmat(sample_path, appendmat = False, variable_names = [self.processed_signals])[self.processed_signals]
        
        user_label = self.user_labels[ind]
        state_label = self.state_labels[ind]
        
        return gram_sample, signal_sample, user_label, state_label
    

if __name__ == '__main__':
    emg = EMG(
            data_dir = '/workspace/processed_data',
            data = pandas.read_hdf('/workspace/processed_data/data_list.hf', key = 'data'),   
            class_num = 3,
            additional_info = True)
    print(len(emg))
    
    def _worker_init_fn(worker_id: int):
        """
        Set numpy seed for the DataLoader worker.
        """
        numpy.random.seed(tor_data.get_worker_info().seed % 1000000000 + worker_id)
    
    data_loader = tor_data.DataLoader(
            emg, batch_size = 16,
            shuffle = True, drop_last = False,
            num_workers = 1, worker_init_fn = _worker_init_fn, pin_memory = False)
    batch = next(iter(data_loader))
    for tensor in batch:
        print(tensor.shape, tensor.dtype)
        print(tensor)
