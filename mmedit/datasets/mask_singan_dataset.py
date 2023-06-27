# Copyright (c) OpenMMLab. All rights reserved.
from copy import deepcopy

import mmcv
import numpy as np
from mmengine.dataset import BaseDataset

from mmedit.registry import DATASETS
import json
import cv2


def splitext(file_name, cl='.'):
    """
    Separate the file name and suffix.
    Args:
        file_name (str): file path
    Returns:
        name (str): file path without extanded name
        suffix (str): extanded name
        e.g. convert 'xx/xx/A.png' to 'xx/xx/A' and '.png'.
    """
    assert isinstance(file_name, str), 'input type must be str.'
    name = file_name
    suffix = ''
    for i in range(len(file_name)-1, -1, -1):
        c = file_name[i]
        if c == cl:
            name = file_name[:i]
            suffix = file_name[i:]
            break
    return name, suffix


def read_json_file(json_path):
    with open(json_path, encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def labelme_polygon_to_mask(points, height, width):
    """
    Args:
        points numpy array (num_points, 2)
    """
    if points.dims < 3:
        points = np.expand_dims(points, 0)
    
    mask = np.zeros((height, width), dtype = "uint8")
    cv2.fillPoly(mask, points, 1)
    return mask


def padding(img, left=0, right=0, top=0, bottom=0, pad=0, value=255):
    """
    Args:
        img (np.array): 
        pad (int): pad size for each egde, if pad != 0
    return:
        pad_img (np.array)
    """
    if pad != 0:
        left = pad
        right = pad
        top = pad
        bottom = pad
    h, w = img.shape[:2]
    new_shape = list(img.shape)
    new_shape[0] = new_shape[0] + top + bottom
    new_shape[1] = new_shape[1] + left + right
    img_new = np.ones(new_shape) * value
    img_new[top:h+top,left:w+left] = img
    return img_new


def create_real_pyramid(real, mask, num_scales, min_size):
    """Create image pyramid.
        padding make it divisible, 
        指定 model 的最大input size 为一个正方形
        现将原图 resize 和 pad 到这个尺寸, 然后生成 图片金字塔
        同时获得最小尺寸的 mask
    This function is modified from the official implementation:
    https://github.com/tamarott/SinGAN/blob/master/SinGAN/functions.py#L221

    In this implementation, we adopt the rescaling function from MMCV.
    Args:
        real (np.array): The real image array.
        mask (np.array): The mask array, same size with image.
        num_scales(int): 
        min_size (int): The minimum size for the image pyramid.
        scale_factor_init (float): The initial scale factor.
    """
    # 如果最终获得输入的尺寸过小，则对原本的 real 图进行 resize
    if min_size < min(real.shape[0], real.shape[1]) / num_scales :

        new_size = min_size * num_scales
        real = mmcv.imrescale(real, new_size / min(real.shape[0], real.shape[1]))
        mask = mmcv.imrescale(mask, new_size / min(real.shape[0], real.shape[1]), interpolation='nearest')

    scale_factor =np.power(min_size / min(real.shape[0], real.shape[1]), 1 / num_scales)
    reals = []
    for i in range(num_scales + 1): # 0, 1, 2, ..., num_scales
        scale = np.power(scale_factor, num_scales - i)
        if i == 0:
            mask = mmcv.imrescale(mask, scale, interpolation="nearest")
        curr_real = mmcv.imrescale(real, scale)
        reals.append(curr_real)

    return reals, scale_factor, mask 


@DATASETS.register_module()
class SinGANDataset(BaseDataset):
    """SinGAN Dataset.

    In this dataset, we create an image pyramid and save it in the cache.

    Args:
        img_path (str): Path to the single image file.
        min_size (int): Min size of the image pyramid. Here, the number will be
            set to the ``min(H, W)``.
        max_size (int): Max size of the image pyramid. Here, the number will be
            set to the ``max(H, W)``.
        scale_factor_init (float): Rescale factor. Note that the actual factor
            we use may be a little bit different from this value.
        num_samples (int, optional): The number of samples (length) in this
            dataset. Defaults to -1.
    """

    def __init__(self,
                 data_root,
                 num_scales,
                 min_size,
                 pipeline,
                 num_samples=-1):
        self.num_scales = num_scales
        self.min_size = min_size
        self.num_samples = num_samples
        super().__init__(data_root=data_root, pipeline=pipeline)

    def full_init(self):
        """Skip the full init process for SinGANDataset.
        Args:
            min_size (int): The minimum size for the image pyramid.
            max_size (int): The maximum size for the image pyramid.
            scale_factor_init (float): The initial scale factor.
        """
        real = mmcv.imread(self.data_root)
        json_path = splitext(self.data_root)[0] + '.json'
        json_data = read_json_file(json_path)
        mask = labelme_polygon_to_mask(json_data["shapes"][0]["points"], height=real.shape[0], width=real.shape[1])
        self.reals, self.scale_factor, self.stop_scale = create_real_pyramid(
            real, mask, self.num_scales, self.min_size)

        self.data_dict = {}

        for i, real in enumerate(self.reals):
            self.data_dict[f'real_scale{i}'] = real

        self.data_dict['input_sample'] = np.zeros_like(
            self.data_dict['real_scale']).astype(np.float32)

    def __getitem__(self, index):
        """Get `:attr:self.data_dict`. For SinGAN, we use single image with
        different resolution to train the model.

        Args:
            idx (int): This will be ignored in :class:`SinGANDataset`.

        Returns:
            dict: Dict contains input image in different resolution.
            ``self.pipeline``.
        """
        return self.pipeline(deepcopy(self.data_dict))

    def __len__(self):
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """
        return int(1e6) if self.num_samples < 0 else self.num_samples
