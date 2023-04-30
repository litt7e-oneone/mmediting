# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import sys
import numpy

import mmcv
import torch
from mmengine import Config, print_log
from mmengine.logging import MMLogger
from mmengine.runner import load_checkpoint, set_random_seed
from functools import reduce
import pickle
# yapf: disable
sys.path.append(os.path.abspath(os.path.join(__file__, '../..')))  # isort:skip  # noqa

from mmedit.engine import *  # isort:skip  # noqa: F401,F403,E402
from mmedit.datasets import *  # isort:skip  # noqa: F401,F403,E402
from mmedit.models import *  # isort:skip  # noqa: F401,F403,E402

from mmedit.registry import MODELS  # isort:skip  # noqa
import cv2
import numpy as np
import random

# yapf: enable
# 变换
def shear(img_grey, alphe=0.2):
    MAS = np.float32([[1, alphe, 0], [0, 1, 0]])  # 构造错切变换矩阵
    imgShear = cv2.warpAffine(img_grey, MAS, 
                              (img_grey.shape[1], img_grey.shape[0]),
                              borderMode=cv2.BORDER_REPLICATE)  # BORDER_REPLICATE  BORDER_TRANSPARENT
    return imgShear

def splitext(file_name):
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
        if c == '.':
            name = file_name[:i]
            suffix = file_name[i:]
            break
    return name, suffix


def get_file_list(source_dir, file_suffix=['.bmp','.BMP','.jpg','.png'], sub_dir=False, only_name=False, all_files=False):
    """
    Gets the path to all files in the root directory with the file_suffix
    Args:

    Returns:
        list ['xx/xx/xx.xx']
    """
    if not (os.path.isdir(source_dir)):
        print(source_dir)
        raise ValueError('The input parameter must be a directory or folder')
    if not (sub_dir == True or sub_dir == False):
        print(sub_dir)
        raise ValueError('The input parameter can only be True or False')
    if not (only_name == True or only_name == False):
        print(only_name)
        raise ValueError('The input parameter can only be True or False')

    if isinstance(file_suffix, str):
        file_suffix = [file_suffix]
    ret = []
    # including all sub-directories
    if sub_dir:
        for root, dirs, files in os.walk(source_dir):
            for name in files:
                if all_files or splitext(name)[-1] in file_suffix:
                    if only_name:
                        ret.append(name)
                    else:
                        ret.append(os.path.join(root, name))

    # not including sub-directories
    else:
        names = os.listdir(source_dir)
        for name in names:
            if  all_files or splitext(name)[-1] in file_suffix:
                if only_name:
                        ret.append(name)
                else:
                    ret.append(os.path.join(source_dir, name))

    if len(ret) == 0:
        print('There is no', file_suffix, 'file in dir', source_dir)
    
    return ret

def back2image(source_img, x1, y1, x2, y2):

    return

def constract_images():
    return

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a GAN model')
    parser.add_argument('--config', default=r'project\singan\collision1\collision1.py', help='evaluation config file path')
    parser.add_argument('--checkpoint', default=r'project\singan\collision1\iter_33000.pth', help='checkpoint file')
    parser.add_argument('--test_pkl_data', default=r'project\singan\collision1\pickle\iter_33001.pkl', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=210, help='random seed')
    parser.add_argument('--task', default='edit', help='test or edit')
    parser.add_argument('--num_samples', default=1000, help='number of random sample')
    # for editing
    parser.add_argument('--image_path', default=r'000000000000200082_4_2_TA07_02_20210908180649902_00_3040_1280_3840_2080-crop.jpg', help='path to iamge to be edited')
    parser.add_argument('--start_stage', default=2, help='start_stage')
    parser.add_argument('--curr_scale', default=10, help='output_scale')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--samples-path',
        type=str,
        default='./result5',  # output dir
        help='path to store images. If not given, remove it after evaluation\
             finished')
    parser.add_argument(
        '--save-prev-res',
        # action='store_true',
        default=False,
        help='whether to store the results from previous stages')
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10,
        help='the number of synthesized samples')
    args = parser.parse_args()
    return args


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

def random_crop_limited(image, target_image, min_ratio=0.6, max_ratio=1, limited=None):  # TODO: 改成装饰器
    """
    limited: [[x1, y1, x2, y2], ...]
    """
    if limited is None:
        limited = [[0, 0, image.shape[1], image.shape[0]]]
    
    if len(limited) == 0:
        limited = [[0, 0, image.shape[1], image.shape[0]]]

    for region in limited:
        x1_t, y1_t, x2_t, y2_t = region
        crop_image, x1, y1, x2, y2 = random_crop(image[y1_t:y2_t, x1_t:x2_t], min_ratio, max_ratio)
        if target_image[y1_t:y2_t, x1_t:x2_t][y1:y2, x1:x2].shape == crop_image.shape:
            print('same')
            target_image[y1_t:y2_t, x1_t:x2_t][y1:y2, x1:x2] = crop_image
    # return target_image

def _tensor2img(img):
    img = img.permute(1, 2, 0)
    # img = ((img + 1) / 2 * 255).clamp(0, 255).to(torch.uint8)
    img = img.clamp(0, 255).to(torch.uint8)
    return img.cpu().numpy()


@torch.no_grad()
def main():
    MMLogger.get_instance('mmedit')

    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.model['test_pkl_data'] = args.test_pkl_data
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # set scope manually
    cfg.model['_scope_'] = 'mmedit'
    # build the model and load checkpoint
    model = MODELS.build(cfg.model)

    model.eval()

    # load ckpt
    print_log(f'Loading ckpt from {args.checkpoint}')
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # add dp wrapper
    if torch.cuda.is_available():
        model = model.cuda()

    for sample_iter in range(args.num_samples):
        if args.task == 'test':
            outputs = model.test_step(
                dict(inputs=dict(num_batches=1, get_prev_res=args.save_prev_res)))

        elif args.task == 'edit':
            with open(args.test_pkl_data, 'rb') as f:
                    pkl_data = pickle.load(f) 
            
            noise = pkl_data['fixed_noises'][args.start_stage]
            pkl_data['fixed_noises'][args.start_stage] = np.random.randn(noise.shape[0],
                                                                        noise.shape[1],
                                                                        noise.shape[2],
                                                                        noise.shape[3] ).astype(np.float32)  # TODO: 添加噪声

            if args.image_path and args.start_stage != 0:
                input_sample = cv2.imread(args.image_path, -1)

                # 取随机样本 TODO: 采样是从 sinGAN随机生成的图中随机采样，限定的区域可以依据 标注缺陷的mask
                sample_pathes = get_file_list(r'D:\Megarobo\mmediting\gen', '.png')
                sample_path = random.choice(sample_pathes)
                sample_image = cv2.imread(sample_path, -1)
                sample_image = cv2.resize(sample_image, (input_sample.shape[1], input_sample.shape[0]))
                # sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
                random_crop_limited(sample_image, input_sample, 0.6, 1, limited = [[0, 0, 800, 26]])  #  [0, 97, 800, 124]
                random_crop_limited(sample_image, input_sample, 0.1, 0.2, limited = [[0, 97, 800, 124]])

                # if input_sample[y1:y2, x1:x2].shape == crop_image.shape:
                #     # print('qa')
                #     input_sample[y1:y2, x1:x2] = crop_image
                # copy random patches
                # TODO:
                # resize 
                image_size = []
                for i, item in enumerate(pkl_data['fixed_noises']):
                    image_size.append((item.shape[-2], item.shape[-1]))
                    # print(i, item.shape)

                h, w = image_size[args.start_stage]

                input_sample = cv2.resize(input_sample, (w, h))

                # normliza
                # channels = input_sample.shape[-1]
                input_sample = np.expand_dims(input_sample, -1)
                input_sample = input_sample.astype(np.float32)
                # 为原图添加变换
                # 随机高斯噪声
                input_sample = input_sample + np.random.randn(input_sample.shape[0],
                                                              input_sample.shape[1],
                                                              input_sample.shape[2],
                                                              ).astype(np.float32)  * np.random.randint(0, 5)   
                # 错切
                # input_sample = shear(input_sample, -0.1)

                


                # mean = np.mean(input_sample)
                # std = np.std(input_sample)
                # input_sample = (input_sample - mean) / std
                input_sample = (input_sample - 127.5) / 127.5
                # input_sample = input_sample / 255

                # channel order to tensor
                if len(input_sample.shape) == 2:
                    input_sample = np.expand_dims(input_sample, -1)
                input_sample = torch.as_tensor(np.ascontiguousarray(input_sample.transpose(2, 0, 1)), dtype=torch.float32)
                # expand dim
                input_sample = input_sample.unsqueeze(0)  # （1, 3, h, w)

                input_sample = input_sample.to(next(model.parameters()).device)

            else: 
                input_sample = None

            outputs = model.edit_step(
                    dict(num_batches=1, get_prev_res=args.save_prev_res, 
                        start_stage=args.start_stage, curr_scale=args.curr_scale, 
                        input_sample=input_sample), pkl_data=pkl_data)
        
        # store results from previous stages
        if args.save_prev_res:
            fake_img = outputs[0].fake_img.data
            prev_res_list = outputs[0].prev_res_list
            prev_res_list.append(fake_img)
            for i, img in enumerate(prev_res_list):
                img = _tensor2img(img)
                # TODO:
                mmcv.imwrite(
                    img,
                    os.path.join(args.samples_path, f'stage{i}',
                                    f'rand_sample_{sample_iter}.png'))
        # just store the final result
        else:
            img = _tensor2img(outputs[0].fake_img.data)
            mmcv.imwrite(
                img,
                os.path.join(args.samples_path,
                                f'rand_sample_{sample_iter}.png'))


if __name__ == '__main__':
    main()
