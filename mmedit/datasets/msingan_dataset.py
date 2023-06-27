from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import mmcv
import os
import numpy as np
from mmengine.dataset import BaseDataset
from mmedit.registry import DATASETS
from .singan_dataset import create_real_pyramid


@DATASETS.register_module()
class MSinGANDataset(BaseDataset):
    """
    Based on SinGAN Dataset.
    support color and grayscale image.

    difference in args data_root, now data_root is the dirctory to images

    In this dataset, we create an image pyramid and save it in the cache.
    """

    def __init__(self,
                 data_root:str,
                 min_size:int,
                 max_size:int,
                 scale_factor_init:float,
                 pipeline:List[Union[dict, Callable]],
                 image_format:str
                 ):
        self.min_size = min_size
        self.max_size = max_size
        self.scale_factor_init = scale_factor_init
        self.image_format = image_format
        super().__init__(data_root=data_root, pipeline=pipeline)
    
    def full_init(self):
        """Skip the full init process for MSinGANDataset, same as SinGANDataset."""

        self.load_data_list(self.min_size, self.max_size,
                            self.scale_factor_init)
    

    def load_data_list(self, min_size, max_size, scale_factor_init):
        """Load annatations for MSinGAN Dataset. Useing for loop.

        Args:
            min_size (int): The minimum size for the image pyramid.
            max_size (int): The maximum size for the image pyramid.
            scale_factor_init (float): The initial scale factor.
        """
        data_names = os.listdir(self.data_root)
        self.data_dict_list = []

        for index, data_name in enumerate(data_names):
            data_path = os.path.join(self.data_root, data_name)

            real = mmcv.imread(data_path, flag=self.image_format)
            self.reals, self.scale_factor, self.stop_scale = create_real_pyramid(
                real, min_size, max_size, scale_factor_init)

            data_dict = {}

            for i, real in enumerate(self.reals):
                data_dict[f'real_scale{i}'] = real

            data_dict['input_sample'] = np.zeros_like(
                data_dict['real_scale0']).astype(np.float32)
            # for constructing fixde noise list
            data_dict['input_index'] = str(index)

            self.data_dict_list.append(data_dict)
        
    
    def __getitem__(self, idx: int) -> dict:
        return self.pipeline(deepcopy(self.data_dict_list[idx]))
    
    def __len__(self):
        return len(self.data_dict_list)
    