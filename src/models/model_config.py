import functools
import dataclasses

from torch import nn as nn
from typing import Dict, List, Tuple, Union


@dataclasses.dataclass
class CnnNetworkConfiguration:
    """Network hyper-parameters."""
    activator_type: str = 'leaky_relu'
    activator_negative_slope: float = 1e-2
    feature_size: int = 64
    feature_extractor: Dict[str, Union[int, List[int], List[Tuple[int, int]], Tuple[int, int]]] = dataclasses.field(
            default_factory = dict(
                    layer = 6,
                    in_channel_num = 1,
                    channel_num = [32, 64, 96, 96, 128, 128],
                    # channel_num = [32, 128],
                    pool_kernel_size = [(3, 3)] * 6,
                    pool_stride = [(1, 1)] * 6,
                    global_pool_channel_num = feature_size,
                    global_pool_out_size = (1, 1)).copy)
    classifier: Dict[str, Union[int, List[int]]] = dataclasses.field(default_factory = dict(
            layer = 3,
            out_size = [256, 256, 3]).copy)
    class_num: int = 3


def get_activator(activator_type: str, alpha: float = 1., negative_slope: float = 1e-2):
    r"""
    
    :param activator_type: Activator type.
    :param alpha: :math:`\alpha` value for the ELU formulation.
    :param negative_slope: `negative_slope` parameter of LeakyReLU.
    :return: The activation function.
    """
    if activator_type == 'elu':
        activator = functools.partial(nn.ELU, alpha = alpha)
    elif activator_type == 'leaky_relu':
        activator = functools.partial(nn.LeakyReLU, negative_slope = negative_slope)
    elif activator_type == 'relu':
        activator = nn.ReLU
    else:
        raise ValueError("Unknown activator type {type} set.".format(type = activator_type))
    return activator

