# -*- coding:utf-8 -*-
import cv2
import math
import numpy as np
from copy import deepcopy 
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
from collections import defaultdict
import random

# 对局部图进行弹性变换等，使其生成随机性的图

def elastic_transform(image, alpha, sigma,
                      alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101) # BORDER_TRANSPARENT 原始值  BORDER_REFLECT_101

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样打的服从[0,1]均匀分布的矩阵
    # *2-1是为了将分布平移到[-1, 1]的区间, alpha是控制变形强度的变形因子
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    # generate meshgrid
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    # x+dx,y+dy
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='nearest').reshape(shape)  
    # mode 
    # ‘reflect’, ‘grid-mirror’, ‘constant’, ‘grid-constant’, ‘nearest’, ‘mirror’, ‘grid-wrap’, ‘wrap’

    return imageC


def affine_transform(image, alpha_affine, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]
    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    # pts1: 仿射变换前的点(3个点)
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size,
                        center_square[1] - square_size],
                       center_square - square_size])
    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                       size=pts1.shape).astype(np.float32)
    # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    # 对image进行仿射变换.
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    return imageB


def shear(img_grey, alphe=0.2):
    MAS = np.float32([[1, alphe, 0], [0, 1, 0]])  # 构造错切变换矩阵
    imgShear = cv2.warpAffine(img_grey, MAS, 
                              (img_grey.shape[1], img_grey.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)  # BORDER_REPLICATE  BORDER_TRANSPARENT
    return imgShear

def random_crop(image, min_ratio=0.6, max_ratio=1):
    h, w = image.shape

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = image[y: y + new_h, x: x+new_w]

    return image, x, y, x + new_w, y + new_h

def random_exchange(image, min_ratio=0.6, max_ratio=1): # 两个相同patch 交换
    w, h = image.shape

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y1 = np.random.randint(0, h - new_h-1)
    x1 = np.random.randint(0, w - new_w-1)

    y2 = np.random.randint(0, h - new_h-1)
    x2 = np.random.randint(0, w - new_w-1)

    temp = image[x1: x1 + new_w, y1: y1+new_h]
    image[x1: x1 + new_w, y1: y1+new_h] = image[x2: x2 + new_w, y2: y2+new_h]
    image[x2: x2 + new_w, y2: y2+new_h] = temp
    return image


# class RandGenVisualization():
#     def __init__(self):
#         self.inputs_buffer = defaultdict(list)
#         return

#     def __call__(self,
#                  module,
#                  ):
#         num_batches = 1
#         module.eval()
#         if hasattr(module, 'module'):
#             module = module.module

#         forward_func = module.val_step

#         for vis_kwargs in self.vis_kwargs_list:
#             # pop the sample-unrelated values
#             vis_kwargs_ = deepcopy(vis_kwargs)
#             sampler_type = vis_kwargs_['type']

#             # replace with alias
#             for alias in self.VIS_KWARGS_MAPPING.keys():
#                 if alias.upper() == sampler_type.upper():
#                     sampler_alias = deepcopy(self.VIS_KWARGS_MAPPING[alias])
#                     vis_kwargs_['type'] = sampler_alias.pop('type')
#                     for default_k, default_v in sampler_alias.items():
#                         vis_kwargs_.setdefault(default_k, default_v)
#                     break
#             # sampler_type = vis_kwargs_.pop('type')

#             name = vis_kwargs_.pop('name', None)
#             if not name:
#                 name = sampler_type.lower()

#             n_samples = vis_kwargs_.pop('n_samples', self.n_samples)
#             n_row = vis_kwargs_.pop('n_row', self.n_row)
#             n_row = min(n_row, n_samples)

#             num_iters = math.ceil(n_samples / num_batches)
#             vis_kwargs_['max_times'] = num_iters
#             vis_kwargs_['num_batches'] = num_batches
#             fixed_input = vis_kwargs_.pop('fixed_input', self.fixed_input)
#             target_keys = vis_kwargs_.pop('target_keys', None)
#             vis_mode = vis_kwargs_.pop('vis_mode', None)

#             output_list = []
#             if fixed_input and self.inputs_buffer[sampler_type]:
#                 sampler = self.inputs_buffer[sampler_type]
#             else:
#                 sampler = get_sampler(vis_kwargs_, runner)
#             need_save = fixed_input and not self.inputs_buffer[sampler_type]

#             for inputs in sampler:
#                 output_list += [out for out in forward_func(inputs)]

#                 # save inputs
#                 if need_save:
#                     self.inputs_buffer[sampler_type].append(inputs)

#             self._visualizer.add_datasample(
#                 name=name,
#                 gen_samples=output_list[:n_samples],
#                 target_keys=target_keys,
#                 vis_mode=vis_mode,
#                 n_row=n_row,
#                 show=self.show,
#                 wait_time=self.wait_time,
#                 step=batch_idx + 1,
#                 **vis_kwargs_)

#         return

if __name__ == '__main__':
    # img_path = '000000000000200082_4_2_TA07_02_20210908180649902_00_3040_1280_3840_2080-crop.jpg'
    # imageA = cv2.imread(img_path)
    # img_show = imageA.copy()
    # imageA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    # # Apply elastic transform on image
    # # imageC = elastic_transform(imageA, imageA.shape[1] * 0.01,
    # #                                imageA.shape[1] * 0.08,
    # #                                imageA.shape[0] * 0.08)

    # # imageC = shear(imageA, -0.4)
    # # image_crop, x1, y1, x2, y2 = random_crop(imageA)
    # # print(image_crop.shape, imageA.shape)
    
    # # imageA[y1:y2, x1: x2] = affine_transform(image_crop, 0.8)
    # # cv2.namedWindow("img_a", 0)
    # # cv2.imshow("img_a", img_show)
    # # cv2.namedWindow("img_c", 0)

    # imageA = random_exchange(imageA)
    # cv2.imshow("img_c", imageA)
    # cv2.waitKey(0)

    region = [1]