import random

import mmcv
import numpy as np
from torch.nn.modules.utils import _pair

from ..registry import PIPELINES
from .utils import random_choose_unknown


@PIPELINES.register_module
class Crop(object):
    """Crop data to specific size for training.

    Attributes:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        random_crop (bool): If set to True, it will random crop
            image. Otherwise, it will work as center crop.
    """

    def __init__(self, keys, crop_size, random_crop=True):
        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')

        self.keys = keys
        self.crop_size = crop_size
        self.random_crop = random_crop

    def _crop(self, data):
        data_h, data_w = data.shape[:2]
        crop_h, crop_w = self.crop_size
        crop_h = min(data_h, crop_h)
        crop_w = min(data_w, crop_w)

        if self.random_crop:
            x_offset = np.random.randint(0, data_w - crop_w + 1)
            y_offset = np.random.randint(0, data_h - crop_h + 1)
        else:
            x_offset = max(0, (data_w - crop_w)) // 2
            y_offset = max(0, (data_h - crop_h)) // 2

        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]

        return data_, crop_bbox

    def __call__(self, results):
        for k in self.keys:
            data_, crop_bbox = self._crop(results[k])
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        results['crop_size'] = self.crop_size
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'random_crop={self.random_crop}')

        return repr_str


@PIPELINES.register_module
class FixedCrop(object):
    """Crop paired data (at a specific position) to specific size for training.

    Attributes:
        keys (Sequence[str]): The images to be cropped.
        crop_size (Tuple[int]): Target spatial size (h, w).
        crop_pos (Tuple[int]): Specific position (x, y). If set to None,
            random initialize the position to crop paired data batch.
    """

    def __init__(self, keys, crop_size, crop_pos=None):
        if not mmcv.is_tuple_of(crop_size, int):
            raise TypeError(
                'Elements of crop_size must be int and crop_size must be'
                f' tuple, but got {type(crop_size[0])} in {type(crop_size)}')
        if not mmcv.is_tuple_of(crop_pos, int) and (crop_pos is not None):
            raise TypeError(
                'Elements of crop_pos must be int and crop_pos must be'
                f' tuple or None, but got {type(crop_pos[0])} in '
                f'{type(crop_pos)}')

        self.keys = keys
        self.crop_size = crop_size
        self.crop_pos = crop_pos

    def _crop(self, data, x_offset, y_offset, crop_w, crop_h):
        crop_bbox = [x_offset, y_offset, crop_w, crop_h]
        data_ = data[y_offset:y_offset + crop_h, x_offset:x_offset + crop_w,
                     ...]
        return data_, crop_bbox

    def __call__(self, results):
        data_h, data_w = results[self.keys[0]].shape[:2]
        crop_h, crop_w = self.crop_size
        crop_h = min(data_h, crop_h)
        crop_w = min(data_w, crop_w)

        if self.crop_pos is None:
            x_offset = np.random.randint(0, data_w - crop_w + 1)
            y_offset = np.random.randint(0, data_h - crop_h + 1)
        else:
            x_offset, y_offset = self.crop_pos
            crop_w = min(data_w - x_offset, crop_w)
            crop_h = min(data_h - y_offset, crop_h)

        for k in self.keys:
            # In fixed crop for paired images, sizes should be the same
            if (results[k].shape[0] != data_h
                    or results[k].shape[1] != data_w):
                raise ValueError(
                    'The sizes of paired images should be the same. Expected '
                    f'({data_h}, {data_w}), but got ({results[k].shape[0]}, '
                    f'{results[k].shape[1]}).')
            data_, crop_bbox = self._crop(results[k], x_offset, y_offset,
                                          crop_w, crop_h)
            results[k] = data_
            results[k + '_crop_bbox'] = crop_bbox
        results['crop_size'] = self.crop_size
        results['crop_pos'] = self.crop_pos
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'keys={self.keys}, crop_size={self.crop_size}, '
                     f'crop_pos={self.crop_pos}')
        return repr_str


@PIPELINES.register_module
class PairedRandomCrop(object):
    """Paried random crop.

    It crops a pair of lq and gt images with corresponding locations.
    It also supports accepting lq list and gt list.
    Required keys are "scale", "lq", and "gt",
    added or modified keys are "lq" and "gt".

    Attributes:
        gt_patch_size (int): cropped gt patch size.
    """

    def __init__(self, gt_patch_size):
        self.gt_patch_size = gt_patch_size

    def __call__(self, results):
        scale = results['scale']
        lq_patch_size = self.gt_patch_size // scale

        lq_is_list = isinstance(results['lq'], list)
        if not lq_is_list:
            results['lq'] = [results['lq']]
        gt_is_list = isinstance(results['gt'], list)
        if not gt_is_list:
            results['gt'] = [results['gt']]

        h_lq, w_lq, _ = results['lq'][0].shape
        h_gt, w_gt, _ = results['gt'][0].shape

        if h_gt != h_lq * scale or w_gt != w_lq * scale:
            raise ValueError(
                f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                f'multiplication of LQ ({h_lq}, {w_lq}).')
        if h_lq < lq_patch_size or w_lq < lq_patch_size:
            raise ValueError(
                f'LQ ({h_lq}, {w_lq}) is smaller than patch size ',
                f'({lq_patch_size}, {lq_patch_size}). Please check '
                f'{results["lq_path"][0]} and {results["gt_path"][0]}.')

        # randomly choose top and left coordinates for lq patch
        top = random.randint(0, h_lq - lq_patch_size)
        left = random.randint(0, w_lq - lq_patch_size)
        # crop lq patch
        results['lq'] = [
            v[top:top + lq_patch_size, left:left + lq_patch_size, ...]
            for v in results['lq']
        ]
        # crop corresponding gt patch
        top_gt, left_gt = int(top * scale), int(left * scale)
        results['gt'] = [
            v[top_gt:top_gt + self.gt_patch_size,
              left_gt:left_gt + self.gt_patch_size, ...] for v in results['gt']
        ]

        if not lq_is_list:
            results['lq'] = results['lq'][0]
        if not gt_is_list:
            results['gt'] = results['gt'][0]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(gt_patch_size={self.gt_patch_size})'
        return repr_str


@PIPELINES.register_module
class CropAroundCenter(object):
    """Randomly crop the images around unknown area in the center 1/4 images.

    This cropping strategy is adopted in GCA matting. The `unknown area` is the
    same as `semi-transparent area`.
    https://arxiv.org/pdf/2001.04069.pdf
    It retains the center 1/4 images and resizes the images to 'crop_size'.
    Required keys are "fg", "bg", "trimap", "alpha", "img_shape" and
    "img_name", modified keys are "fg", "bg", "trimap", "alpha" and
    "img_shape".

    Attributes:
        crop_size (int | tuple): Desired output size. If int, square crop is
            applied.
    """

    def __init__(self, crop_size):
        if mmcv.is_tuple_of(crop_size, int):
            assert len(crop_size) == 2, 'length of crop_size must be 2.'
        elif not isinstance(crop_size, int):
            raise TypeError('crop_size must be int or a tuple of int, but got '
                            f'{type(crop_size)}')
        self.crop_size = _pair(crop_size)

    def __call__(self, results):
        fg = results['fg']
        alpha = results['alpha']
        trimap = results['trimap']
        bg = results['bg']
        h, w = results['img_shape'][:2]
        assert bg.shape == fg.shape, (f'shape of bg {bg.shape} should be the '
                                      f'same as fg {fg.shape}.')

        crop_h, crop_w = self.crop_size
        # Make sure h >= crop_h, w >= crop_w. If not, rescale imgs
        rescale_ratio = max(crop_h / h, crop_w / w)
        if rescale_ratio > 1:
            new_h = max(int(h * rescale_ratio), crop_h)
            new_w = max(int(w * rescale_ratio), crop_w)
            fg = mmcv.imresize(fg, (new_w, new_h), interpolation='nearest')
            alpha = mmcv.imresize(
                alpha, (new_w, new_h), interpolation='nearest')
            trimap = mmcv.imresize(
                trimap, (new_w, new_h), interpolation='nearest')
            bg = mmcv.imresize(bg, (new_w, new_h), interpolation='bicubic')
            h, w = new_h, new_w

        # resize to 1/4 to ignore small unknown patches
        small_trimap = mmcv.imresize(
            trimap, (w // 4, h // 4), interpolation='nearest')
        # find unknown area in center 1/4 region
        margin_h, margin_w = crop_h // 2, crop_w // 2
        sample_area = small_trimap[margin_h // 4:(h - margin_h) // 4,
                                   margin_w // 4:(w - margin_w) // 4]
        unknown_xs, unknown_ys = np.where(sample_area == 128)
        unknown_num = len(unknown_xs)
        if unknown_num < 10:
            # too few unknown area in the center, crop from the whole image
            top = np.random.randint(0, h - crop_h + 1)
            left = np.random.randint(0, w - crop_w + 1)
        else:
            idx = np.random.randint(unknown_num)
            top = unknown_xs[idx] * 4
            left = unknown_ys[idx] * 4
        bottom = top + crop_h
        right = left + crop_w

        results['fg'] = fg[top:bottom, left:right]
        results['alpha'] = alpha[top:bottom, left:right]
        results['trimap'] = trimap[top:bottom, left:right]
        results['bg'] = bg[top:bottom, left:right]
        results['img_shape'] = (crop_h, crop_w)
        results['crop_bbox'] = (left, top, right, bottom)
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module
class CropAroundUnknown(object):
    """Crop around unknown area with a randomly selected scale.

    Randomly select the w and h from a list of (w, h).
    Required keys are "img_shape" and the keys in argument `keys`, added or
    modified keys are "img_shape", "crop_bbox" and the keys in argument `keys`.
    This class assumes value of "alpha" ranges from 0 to 255.

    Attributes:
        keys (Sequence[str]): The images to be cropped. It must contain
            'alpha'. If unknown_source is set to 'trimap', then it must also
            contain 'trimap'.
        crop_sizes (list[int | tuple[int]]): List of (w, h) to be selected.
        unknown_source (str, optional): Unknown area to select from. It must be
            'alpha' or 'tirmap'. Default to 'alpha'.
        interpolation (str, optional): Interpolation method of mmcv.imresize.
            The interpolation operation will be applied when image size is
            smaller than the crop_size. Default to 'bilinear'.
    """

    def __init__(self,
                 keys,
                 crop_sizes,
                 unknown_source='alpha',
                 interpolation='bilinear'):
        if 'alpha' not in keys:
            raise ValueError(f'"alpha" must be in keys, but got {keys}')
        self.keys = keys

        if not isinstance(crop_sizes, list):
            raise TypeError(
                f'Crop sizes must be list, but got {type(crop_sizes)}.')
        self.crop_sizes = [_pair(crop_size) for crop_size in crop_sizes]
        if not mmcv.is_tuple_of(self.crop_sizes[0], int):
            raise TypeError('Elements of crop_sizes must be int or tuple of '
                            f'int, but got {type(self.crop_sizes[0][0])}.')

        if unknown_source not in ['alpha', 'trimap']:
            raise ValueError('unknown_source must be "alpha" or "trimap", '
                             f'but got {unknown_source}')
        elif unknown_source not in keys:
            # it could only be trimap, since alpha is checked before
            raise ValueError(
                'if unknown_source is "trimap", it must also be set in keys')
        self.unknown_source = unknown_source

        self.interpolation = interpolation

    def __call__(self, results):
        h, w = results['img_shape'][:2]

        rand_ind = np.random.randint(len(self.crop_sizes))
        crop_h, crop_w = self.crop_sizes[rand_ind]

        # Make sure h >= crop_h, w >= crop_w. If not, rescale imgs
        rescale_ratio = max(crop_h / h, crop_w / w)
        if rescale_ratio > 1:
            h = max(int(h * rescale_ratio), crop_h)
            w = max(int(w * rescale_ratio), crop_w)
            for key in self.keys:
                results[key] = mmcv.imresize(
                    results[key], (w, h), interpolation=self.interpolation)

        # Select the cropping top-left point which is an unknown pixel
        if self.unknown_source == 'alpha':
            unknown = (results['alpha'] > 0) & (results['alpha'] < 255)
        else:
            unknown = results['trimap'] == 128
        top, left = random_choose_unknown(unknown, (crop_h, crop_w))

        bottom = top + crop_h
        right = left + crop_w

        for key in self.keys:
            results[key] = results[key][top:bottom, left:right]
        results['img_shape'] = results['alpha'].shape
        results['crop_bbox'] = (left, top, right, bottom)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(keys={self.keys}, crop_sizes={self.crop_sizes}, '
                     f"unknown_source='{self.unknown_source}', "
                     f"interpolation='{self.interpolation}')")
        return repr_str


@PIPELINES.register_module
class ModCrop(object):
    """Mod crop gt images, used during testing.

    Required keys are "scale" and "gt",
    added or modified keys are "gt".
    """

    def __call__(self, results):
        img = results['gt'].copy()
        scale = results['scale']
        if img.ndim in [2, 3]:
            h, w = img.shape[0], img.shape[1]
            h_remainder, w_remainder = h % scale, w % scale
            img = img[:h - h_remainder, :w - w_remainder, ...]
        else:
            raise ValueError(f'Wrong img ndim: {img.ndim}.')
        results['gt'] = img
        return results
