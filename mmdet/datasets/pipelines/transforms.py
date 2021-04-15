import inspect

import mmcv
import cv2
import numpy as np
from numpy import random
from PIL import Image

from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
from ..registry import PIPELINES

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

MAX_VALUES_BY_DTYPE = {np.dtype("uint8"):255, np.dtype("uint16"):65535, np.dtype("uint32"):4294967295,np.dtype("float32"):1.0}

@PIPELINES.register_module
class Resize(object):
    """Resize images & bbox & mask.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. If the input dict contains the key
    "scale", then the scale in the input dict is used, otherwise the specified
    scale in the init method is used.

    `img_scale` can either be a tuple (single-scale) or a list of tuple
    (multi-scale). There are 3 multiscale modes:
    - `ratio_range` is not None: randomly sample a ratio from the ratio range
        and multiply it with the image scale.
    - `ratio_range` is None and `multiscale_mode` == "range": randomly sample a
        scale from the a range.
    - `ratio_range` is None and `multiscale_mode` == "value": randomly sample a
        scale from multiple scales.

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        if self.ratio_range is not None:
            scale, scale_idx = self.random_sample_ratio(
                self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
            scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                    dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_score_maps(self, results):
        score_maps = []
        for img in results['score_maps']:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(img, results['scale'], return_scale=True)
            else:
                img, w_scale, h_scale = mmcv.imresize(img, results['scale'], return_scale=True)
            score_maps.append(img)
        results['score_maps'] = score_maps

    def _resize_training_mask(self, results):
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(results['training_mask'], results['scale'], return_scale=True)
        else:
            img, w_scale, h_scale = mmcv.imresize(results['training_mask'], results['scale'], return_scale=True)

        results['training_mask'] = img

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    mmcv.imrescale(
                        mask, results['scale_factor'], interpolation='nearest')
                    for mask in results[key]
                ]
            else:
                mask_size = (results['img_shape'][1], results['img_shape'][0])
                masks = [
                    mmcv.imresize(mask, mask_size, interpolation='nearest')
                    for mask in results[key]
                ]
            results[key] = np.stack(masks)

    def _resize_seg(self, results):
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results['gt_semantic_seg'] = gt_seg

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        if 'training_mask' in results.keys():
            self._resize_training_mask(results)
        if 'score_maps' in  results.keys():
            self._resize_score_maps(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(img_scale={}, multiscale_mode={}, ratio_range={}, '
                     'keep_ratio={})').format(self.img_scale,
                                              self.multiscale_mode,
                                              self.ratio_range,
                                              self.keep_ratio)
        return repr_str


@PIPELINES.register_module
class RandomFlip(object):
    """Flip the image & bbox & mask.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        flip_ratio (float, optional): The flipping probability.
    """

    def __init__(self, flip_ratio=None, direction='horizontal'):
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally.

        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key],
                                              results['img_shape'],
                                              results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = np.stack([
                    mmcv.imflip(mask, direction=results['flip_direction'])
                    for mask in results[key]
                ])

            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imflip(
                    results[key], direction=results['flip_direction'])

            #flip score_maps
            if 'score_maps' in results.keys():
                score_maps = []
                for score_map in results['score_maps']:
                    score_map = mmcv.imflip(score_map, direction=results['flip_direction'])
                    score_maps.append(score_map)
                results['score_maps'] = score_maps

            # flip training_mask
            if 'training_mask' in results.keys():
                results['training_mask'] = mmcv.imflip(results['training_mask'], direction=results['flip_direction'])

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


@PIPELINES.register_module
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value, 0 by default.
    """

    def __init__(self, size=None, size_divisor=None, pad_val=0):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['img'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['img_shape'] = padded_img.shape
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_score_maps(self, results):
        score_maps = []
        for score_map in results['score_maps']:
            if self.size is not None:
                padded_img = mmcv.impad(score_map, self.size, self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(score_map, self.size_divisor, pad_val=self.pad_val)
            score_maps.append(padded_img)
        results['score_maps'] = score_maps

    def _pad_training_mask(self, results):
        if self.size is not None:
            padded_img = mmcv.impad(results['training_mask'], self.size, self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['training_mask'], self.size_divisor, pad_val=self.pad_val)
        results['training_mask'] = padded_img

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            padded_masks = [
                mmcv.impad(mask, pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            if padded_masks:
                results[key] = np.stack(padded_masks, axis=0)
            else:
                results[key] = np.empty((0, ) + pad_shape, dtype=np.uint8)

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(results[key], results['pad_shape'][:2])

    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        if 'score_maps' in results.keys():
            self._pad_score_maps(results)
        if 'training_mask' in results.keys():
            self._pad_training_mask(results)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


@PIPELINES.register_module
class Normalize(object):
    """Normalize the image.

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, std={}, to_rgb={})'.format(
            self.mean, self.std, self.to_rgb)
        return repr_str

@PIPELINES.register_module
class Randomrotate(object):
    def __init__(self, angle_range, rotate_ratio=0.5, mode='random'):
        rotate_angle_list = angle_range if isinstance(angle_range, list) else [angle_range]

        if mode == 'random':
            if len(rotate_angle_list) == 1:
                self.rot_ang = [-np.abs(rotate_angle_list[0]), np.abs(rotate_angle_list[0])]
            elif len(rotate_angle_list) == 2:
                self.rot_ang = [min(rotate_angle_list), max(rotate_angle_list)]
            else:
                raise ValueError('tor_ang must be a value or two_value_list')
        elif mode == 'fix':
            self.rot_ang = rotate_angle_list
        else:
            raise ValueError('rotate mode need to be fix or random')

        self.p = rotate_ratio
        self.mode = mode

    def _rotate_poly(self, polys, M):
        new_polys = list(polys)
        for i , coord in enumerate(new_polys):
            v = [coord[0], coord[1], 1]
            calculated = np.dot(M, v)
            new_polys[i] = (calculated[0], calculated[1])
        return new_polys

    def __call__(self, results):
        if np.random.uniform(0,1) < self.p:
            img = results['img']
            (h, w) = img.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            if self.mode == 'random':
                angle = np.random.uniform(self.rot_ang[0], self.rot_ang[1])
            else:
                angle = random.choice(self.rot_ang)

            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            cos = np.abs(M[0,0])
            sin = np.abs(M[0,1])

            nW = int((h*sin) + (w*cos))
            nH = int((h * cos) + (w * sin))

            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            rotate_img = cv2.warpAffine(img, M, (nW,nH))
            img_shape = rotate_img.shape
            results['img'] = rotate_img
            results['img_shape'] = img_shape
            results['pad_shape'] = img_shape

            if 'score_maps' in results:
                score_maps = []
                for score_map in results['score_maps']:
                    gt_text = cv2.warpAffine(score_map.astype('uint8'), M, (nW, nH))
                    score_maps.append(gt_text)
                results['score_maps'] = score_maps

            if 'training_mask' in results:
                valid_gt_masks = []
                training_mask = cv2.warpAffine(results['training_mask'].astype('uint8'), M, (nW, nH))
                results['training_mask'] = training_mask

            # rotate bboxes accordingly and clip to the image boundary
            for key in results.get('bbox_fields', []):
                if key == 'gt_bboxes' or key == 'gt_bboxes_ignore':
                    num_box = results[key].shape[0]
                    v1 = np.hstack((results[key][:, 0], results[key][:, 1], np.ones(num_box))).reshape(3, num_box)
                    v2 = np.hstack((results[key][:, 2], results[key][:, 1], np.ones(num_box))).reshape(3, num_box)
                    v3 = np.hstack((results[key][:, 2], results[key][:, 3], np.ones(num_box))).reshape(3, num_box)
                    v4 = np.hstack((results[key][:, 0], results[key][:, 3], np.ones(num_box))).reshape(3, num_box)
                    rotate1 = np.dot(M, v1)
                    rotate2 = np.dot(M, v2)
                    rotate3 = np.dot(M, v3)
                    rotate4 = np.dot(M, v4)
                    concat = np.vstack((rotate1, rotate2, rotate3, rotate4))
                    concat = concat.astype(np.int32)
                    for i in range(num_box):
                        rx, ry, rw, rh = cv2.boundingRect(concat[:, i].reshape(-1, 2))
                        rx_min = rx
                        ry_min = ry
                        rx_max = rx + rw
                        ry_max = ry + rh
                        results[key][i, :] = np.array([rx_min, ry_min, rx_max, ry_max])
                    results[key][:, 0::2] = np.clip(results[key][:, 0::2], 0, results['img_shape'][1] - 1)
                    results[key][:, 1::2] = np.clip(results[key][:, 1::2], 0, results['img_shape'][0] - 1)
                elif key == 'gt_polys':
                    num_poly = len(results[key])
                    rotate_polys = []
                    for polys in results[key]:
                        new_poly = np.array(self._roatte_polys(polys, M))
                        new_poly[:, 0] = np.clip(new_poly[:, 0], 0, results['img_shape'][1] - 1)
                        new_poly[:, 1] = np.clip(new_poly[:, 1], 0, results['img_shape'][0] - 1)
                        rotate_polys.append(new_poly)
                    results[key] = rotate_polys

            # filter out the gt bboxes that are completely cropped
            if 'gt_bboxes' in results:
                gt_bboxes = results['gt_bboxes']
                valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                        gt_bboxes[:, 3] > gt_bboxes[:, 1])
                # if no gt bbox remains after cropping, just skip this image
                if not np.any(valid_inds):
                    return None
                results['gt_bboxes'] = gt_bboxes[valid_inds, :]
                if 'gt_polys' in results:
                    valid_polys = []
                    for flag, poly in zip(valid_inds, results['gt_polys']):
                        if flag:
                            valid_polys.append(poly)
                    results['gt_polys'] = valid_polys
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid_inds]

                # filter and crop the masks
                if 'gt_masks' in results:
                    valid_gt_masks = []
                    for i in np.where(valid_inds)[0]:
                        gt_mask = cv2.warpAffine(results['gt_masks'][i], M,
                                                 (nW, nH))
                        valid_gt_masks.append(gt_mask)
                    results['gt_masks'] = valid_gt_masks

                if 'gt_semantic_seg' in results:
                    gt_text = cv2.warpAffine(results['gt_semantic_seg'].astype('uint8'), M,
                                             (nW, nH))
                    results['gt_semantic_seg'] = gt_text

            if 'gt_bboxes' in results:
                gt_bboxes = results['gt_bboxes']
                if gt_bboxes.shape[0] == 0:
                    return None
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(rotate_angle={})'.format(
            self.rot_ang)


@PIPELINES.register_module
class RandomCrop(object):
    """Random crop the image & bboxes & masks.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
    """

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        img = results['img']
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['pad_shape'] = img_shape

        if 'score_maps' in results:
            score_maps = []
            for score_map in results['score_maps']:
                score_map = score_map[crop_y1:crop_y2, crop_x1:crop_x2, ...]
                score_maps.append(score_map)
            results['score_maps'] = score_maps

        if 'training_mask' in results:
            results['training_mask'] = results['training_mask'][crop_y1:crop_y2, crop_x1:crop_x2, ...]

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32)
            bboxes = results[key] - bbox_offset
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = results[key][crop_y1:crop_y2, crop_x1:crop_x2]

        # filter out the gt bboxes that are completely cropped
        if 'gt_bboxes' in results:
            gt_bboxes = results['gt_bboxes']
            valid_inds = (gt_bboxes[:, 2] > gt_bboxes[:, 0]) & (
                gt_bboxes[:, 3] > gt_bboxes[:, 1])
            # if no gt bbox remains after cropping, just skip this image
            if not np.any(valid_inds):
                return None
            results['gt_bboxes'] = gt_bboxes[valid_inds, :]
            if 'gt_labels' in results:
                results['gt_labels'] = results['gt_labels'][valid_inds]

            # filter and crop the masks
            if 'gt_masks' in results:
                valid_gt_masks = []
                for i in np.where(valid_inds)[0]:
                    gt_mask = results['gt_masks'][i][crop_y1:crop_y2,
                                                     crop_x1:crop_x2]
                    valid_gt_masks.append(gt_mask)
                results['gt_masks'] = np.stack(valid_gt_masks)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(
            self.crop_size)


@PIPELINES.register_module
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(scale_factor={})'.format(
            self.scale_factor)


@PIPELINES.register_module
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        img = results['img']
        # random brightness
        if random.randint(2):
            delta = random.uniform(-self.brightness_delta,
                                   self.brightness_delta)
            img += delta

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # convert color from BGR to HSV
        img = mmcv.bgr2hsv(img)

        # random saturation
        if random.randint(2):
            img[..., 1] *= random.uniform(self.saturation_lower,
                                          self.saturation_upper)

        # random hue
        if random.randint(2):
            img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
            img[..., 0][img[..., 0] > 360] -= 360
            img[..., 0][img[..., 0] < 0] += 360

        # convert color from HSV to BGR
        img = mmcv.hsv2bgr(img)

        # random contrast
        if mode == 0:
            if random.randint(2):
                alpha = random.uniform(self.contrast_lower,
                                       self.contrast_upper)
                img *= alpha

        # randomly swap channels
        if random.randint(2):
            img = img[..., random.permutation(3)]

        results['img'] = img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(brightness_delta={}, contrast_range={}, '
                     'saturation_range={}, hue_delta={})').format(
                         self.brightness_delta, self.contrast_range,
                         self.saturation_range, self.hue_delta)
        return repr_str


@PIPELINES.register_module
class Expand(object):
    """Random expand the image & bboxes.

    Randomly place the original image on a canvas of 'ratio' x original image
    size filled with mean values. The ratio is in the range of ratio_range.

    Args:
        mean (tuple): mean value of dataset.
        to_rgb (bool): if need to convert the order of mean to align with RGB.
        ratio_range (tuple): range of expand ratio.
        prob (float): probability of applying this transformation
    """

    def __init__(self,
                 mean=(0, 0, 0),
                 to_rgb=True,
                 ratio_range=(1, 4),
                 seg_ignore_label=None,
                 prob=0.5):
        self.to_rgb = to_rgb
        self.ratio_range = ratio_range
        if to_rgb:
            self.mean = mean[::-1]
        else:
            self.mean = mean
        self.min_ratio, self.max_ratio = ratio_range
        self.seg_ignore_label = seg_ignore_label
        self.prob = prob

    def __call__(self, results):
        if random.uniform(0, 1) > self.prob:
            return results

        img, boxes = [results[k] for k in ('img', 'gt_bboxes')]

        h, w, c = img.shape
        ratio = random.uniform(self.min_ratio, self.max_ratio)
        expand_img = np.full((int(h * ratio), int(w * ratio), c),
                             self.mean).astype(img.dtype)
        left = int(random.uniform(0, w * ratio - w))
        top = int(random.uniform(0, h * ratio - h))
        expand_img[top:top + h, left:left + w] = img
        boxes = boxes + np.tile((left, top), 2).astype(boxes.dtype)

        results['img'] = expand_img
        results['gt_bboxes'] = boxes

        if 'gt_masks' in results:
            expand_gt_masks = []
            for mask in results['gt_masks']:
                expand_mask = np.full((int(h * ratio), int(w * ratio)),
                                      0).astype(mask.dtype)
                expand_mask[top:top + h, left:left + w] = mask
                expand_gt_masks.append(expand_mask)
            results['gt_masks'] = np.stack(expand_gt_masks)

        # not tested
        if 'gt_semantic_seg' in results:
            assert self.seg_ignore_label is not None
            gt_seg = results['gt_semantic_seg']
            expand_gt_seg = np.full((int(h * ratio), int(w * ratio)),
                                    self.seg_ignore_label).astype(gt_seg.dtype)
            expand_gt_seg[top:top + h, left:left + w] = gt_seg
            results['gt_semantic_seg'] = expand_gt_seg
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(mean={}, to_rgb={}, ratio_range={}, ' \
                    'seg_ignore_label={})'.format(
                        self.mean, self.to_rgb, self.ratio_range,
                        self.seg_ignore_label)
        return repr_str


@PIPELINES.register_module
class MinIoURandomCrop(object):
    """Random crop the image & bboxes, the cropped patches have minimum IoU
    requirement with original image & bboxes, the IoU threshold is randomly
    selected from min_ious.

    Args:
        min_ious (tuple): minimum IoU threshold for all intersections with
        bounding boxes
        min_crop_size (float): minimum crop's size (i.e. h,w := a*h, a*w,
        where a >= min_crop_size).
    """

    def __init__(self, min_ious=(0.1, 0.3, 0.5, 0.7, 0.9), min_crop_size=0.3):
        # 1: return ori img
        self.sample_mode = (1, *min_ious, 0)
        self.min_crop_size = min_crop_size

    def __call__(self, results):
        img, boxes, labels = [
            results[k] for k in ('img', 'gt_bboxes', 'gt_labels')
        ]
        h, w, c = img.shape
        while True:
            mode = random.choice(self.sample_mode)
            if mode == 1:
                return results

            min_iou = mode
            for i in range(50):
                new_w = random.uniform(self.min_crop_size * w, w)
                new_h = random.uniform(self.min_crop_size * h, h)

                # h / w in [0.5, 2]
                if new_h / new_w < 0.5 or new_h / new_w > 2:
                    continue

                left = random.uniform(w - new_w)
                top = random.uniform(h - new_h)

                patch = np.array(
                    (int(left), int(top), int(left + new_w), int(top + new_h)))
                overlaps = bbox_overlaps(
                    patch.reshape(-1, 4), boxes.reshape(-1, 4)).reshape(-1)
                if overlaps.min() < min_iou:
                    continue

                # center of boxes should inside the crop img
                center = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = ((center[:, 0] > patch[0]) * (center[:, 1] > patch[1]) *
                        (center[:, 0] < patch[2]) * (center[:, 1] < patch[3]))
                if not mask.any():
                    continue
                boxes = boxes[mask]
                labels = labels[mask]

                # adjust boxes
                img = img[patch[1]:patch[3], patch[0]:patch[2]]
                boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
                boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
                boxes -= np.tile(patch[:2], 2)

                results['img'] = img
                results['gt_bboxes'] = boxes
                results['gt_labels'] = labels

                if 'gt_masks' in results:
                    valid_masks = [
                        results['gt_masks'][i] for i in range(len(mask))
                        if mask[i]
                    ]
                    results['gt_masks'] = np.stack([
                        gt_mask[patch[1]:patch[3], patch[0]:patch[2]]
                        for gt_mask in valid_masks
                    ])

                # not tested
                if 'gt_semantic_seg' in results:
                    results['gt_semantic_seg'] = results['gt_semantic_seg'][
                        patch[1]:patch[3], patch[0]:patch[2]]
                return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(min_ious={}, min_crop_size={})'.format(
            self.min_ious, self.min_crop_size)
        return repr_str


@PIPELINES.register_module
class Corrupt(object):

    def __init__(self, corruption, severity=1):
        self.corruption = corruption
        self.severity = severity

    def __call__(self, results):
        if corrupt is None:
            raise RuntimeError('imagecorruptions is not installed')
        results['img'] = corrupt(
            results['img'].astype(np.uint8),
            corruption_name=self.corruption,
            severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(corruption={}, severity={})'.format(
            self.corruption, self.severity)
        return repr_str


@PIPELINES.register_module
class Albu(object):

    def __init__(self,
                 transforms,
                 bbox_params=None,
                 keymap=None,
                 update_pad_shape=False,
                 skip_img_without_anno=False):
        """
        Adds custom transformations from Albumentations lib.
        Please, visit `https://albumentations.readthedocs.io`
        to get more information.

        transforms (list): list of albu transformations
        bbox_params (dict): bbox_params for albumentation `Compose`
        keymap (dict): contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): whether to skip the image
                                      if no ann left after aug
        """
        if Compose is None:
            raise RuntimeError('albumentations is not installed')

        self.transforms = transforms
        self.filter_lost_elements = False
        self.update_pad_shape = update_pad_shape
        self.skip_img_without_anno = skip_img_without_anno

        # A simple workaround to remove masks without boxes
        if (isinstance(bbox_params, dict) and 'label_fields' in bbox_params
                and 'filter_lost_elements' in bbox_params):
            self.filter_lost_elements = True
            self.origin_label_fields = bbox_params['label_fields']
            bbox_params['label_fields'] = ['idx_mapper']
            del bbox_params['filter_lost_elements']

        self.bbox_params = (
            self.albu_builder(bbox_params) if bbox_params else None)
        self.aug = Compose([self.albu_builder(t) for t in self.transforms],
                           bbox_params=self.bbox_params)

        if not keymap:
            self.keymap_to_albu = {
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            }
        else:
            self.keymap_to_albu = keymap
        self.keymap_back = {v: k for k, v in self.keymap_to_albu.items()}

    def albu_builder(self, cfg):
        """Import a module from albumentations.
        Inherits some of `build_from_cfg` logic.

        Args:
            cfg (dict): Config dict. It should at least contain the key "type".
        Returns:
            obj: The constructed object.
        """
        assert isinstance(cfg, dict) and "type" in cfg
        args = cfg.copy()

        obj_type = args.pop("type")
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                'type must be a str or valid type, but got {}'.format(
                    type(obj_type)))

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    @staticmethod
    def mapper(d, keymap):
        """
        Dictionary mapper.
        Renames keys according to keymap provided.

        Args:
            d (dict): old dict
            keymap (dict): {'old_key':'new_key'}
        Returns:
            dict: new dict.
        """
        updated_dict = {}
        for k, v in zip(d.keys(), d.values()):
            new_k = keymap.get(k, k)
            updated_dict[new_k] = d[k]
        return updated_dict

    def __call__(self, results):
        # dict to albumentations format
        results = self.mapper(results, self.keymap_to_albu)

        if 'bboxes' in results:
            # to list of boxes
            if isinstance(results['bboxes'], np.ndarray):
                results['bboxes'] = [x for x in results['bboxes']]
            # add pseudo-field for filtration
            if self.filter_lost_elements:
                results['idx_mapper'] = np.arange(len(results['bboxes']))

        results = self.aug(**results)

        if 'bboxes' in results:
            if isinstance(results['bboxes'], list):
                results['bboxes'] = np.array(
                    results['bboxes'], dtype=np.float32)
            results['bboxes'] = results['bboxes'].reshape(-1, 4)

            # filter label_fields
            if self.filter_lost_elements:

                results['idx_mapper'] = np.arange(len(results['bboxes']))

                for label in self.origin_label_fields:
                    results[label] = np.array(
                        [results[label][i] for i in results['idx_mapper']])
                if 'masks' in results:
                    results['masks'] = np.array(
                        [results['masks'][i] for i in results['idx_mapper']])

                if (not len(results['idx_mapper'])
                        and self.skip_img_without_anno):
                    return None

        if 'gt_labels' in results:
            if isinstance(results['gt_labels'], list):
                results['gt_labels'] = np.array(results['gt_labels'])
            results['gt_labels'] = results['gt_labels'].astype(np.int64)

        # back to the original format
        results = self.mapper(results, self.keymap_back)

        # update final shape
        if self.update_pad_shape:
            results['pad_shape'] = results['img'].shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transformations={})'.format(self.transformations)
        return repr_str

@PIPELINES.register_module
class RandomBlur(object):
    def __init__(self, ksize=3, p=1.0):
        self.ksize = ksize
        self.p = p

    def __call__(self, results):
        if random.random() < self.p:
            img = results['img']
            tp = random.random()
            if tp < 0.3:
                resultImg = cv2.medianBlur(img, self.ksize)
                results['img'] = resultImg
            elif tp < 0.6:
                sigma = 0
                k_list = [self.ksize,self.ksize]
                kw = (k_list[0] * 2) + 1
                kh = (k_list[1] * 2) + 1
                resultImg = cv2.GaussianBlur(img, (kw, kh), sigma)
                results['img'] = resultImg
            else:
                k_list = (self.ksize,self.ksize)
                resultImg = cv2.blur(img, k_list)
                results['img'] = resultImg
        return results

def adjust_Brightness(img, factor):
    if factor == 0:
        return np.zeros_like(img)
    elif factor == 1:
        return img

    if img.dtype == np.uint8:
        lut = np.arange(0, 256) * factor
        lut = np.clip(lut, 0, 255).astype(np.uint8)
        return cv2.LUT(img, lut)

    return img

def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)

def adjust_Contrast(img, factor):
    mean = None
    if factor == 1:
        return img

    if (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1):
        mean = img.mean()
    else:
        mean = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).mean()

    if factor == 0:
        return np.full_like(img, int(mean + 0.5), dtype=img.dtype)

    if img.dtype == np.uint8:
        lut = np.arange(0, 256) * factor
        lut = lut + mean * (1 - factor)
        lut = clip(lut, img.dtype, 255)
        return cv2.LUT(img, lut)

    return clip(img.astype(np.float32) * factor + mean * (1 - factor), img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])


def adjust_Saturation(img, factor, gamma=0):
    gray = None
    if factor == 1:
        return img

    if (len(img.shape) == 2) or (len(img.shape) == 3 and img.shape[-1] == 1):
        gray = img
        return gray
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    if factor == 0:
        return gray

    result = cv2.addWeighted(img, factor, gray, 1 - factor, gamma=gamma)

    if img.dtype == np.uint8:
        return result

    return clip(result, img.dtype, MAX_VALUES_BY_DTYPE[img.dtype])

@PIPELINES.register_module
class BrightnessContrastSaturation(object):
    def __init__(self, brightness_range=[0.5, 1.3], contrast_range=[0.5, 1.5], saturation_range=[0.5, 1.5], p=1.0):
        self.brightness_lower = brightness_range[0]
        self.brightness_upper = brightness_range[1]
        self.contrast_lower = contrast_range[0]
        self.contrast_upper = contrast_range[1]
        self.saturation_lower = saturation_range[0]
        self.saturation_upper = saturation_range[1]
        self.p = p

    def __call__(self, results):
        if random.random() < self.p:
            color_mode = random.random()#random.randint(0, 2)
            img = results['img']
            if color_mode < 0.4:#== 0:
                factor = random.uniform(self.brightness_lower, self.brightness_upper)
                img = adjust_Brightness(img, factor)

            elif color_mode < 0.8:#== 1:
                factor = random.uniform(self.contrast_lower, self.contrast_upper)
                img = adjust_Contrast(img, factor)

            else:
                factor = random.uniform(self.saturation_lower, self.saturation_upper)
                img = adjust_Saturation(img, factor)

            results['img'] = img
        return results

def to_float(img, max_value=None):
    if max_value is None:
        max_value = MAX_VALUES_BY_DTYPE[img.dtype]
    return img.astype("float32") / max_value


def from_float(img, dtype, max_value=None):
    if max_value is None:
        max_value = MAX_VALUES_BY_DTYPE[dtype]
    return (img * max_value).astype(dtype)


@PIPELINES.register_module
class RandomImgCompression(object):
    def __init__(self, quality=[85, 95], p=1.0):
        self.quality = quality
        self.p = p

    def __call__(self, results):
        if random.random() < self.p:
            quality = random.randint(self.quality[0], self.quality[1])
            img = results['img']
            img_type = '.' + results['filename'].split('.')[-1]
            if img_type not in ['.jpg', '.jpeg']:
                return results
            quality_flag = cv2.IMWRITE_JPEG_QUALITY
            input_type = img.dtype
            need_float = False
            if input_type == np.float32:
                img = from_float(img, dtype=np.dtype("uint8"))
                need_float = True
            elif input_type not in (np.uint8, np.float32):
                return results

            _, encode_img = cv2.imencode(img_type, img, (int(quality_flag), quality))
            img = cv2.imdecode(encode_img, cv2.IMREAD_UNCHANGED)
            if need_float:
                img = to_float(img, max_value=255)
            results['img'] = img
        return results

@PIPELINES.register_module
class Resize_SAR(object):
    def __init__(self, img_scale=(48,256)):
        self.h, self.w = img_scale[0], img_scale[1]
        self.interpolation = Image.BILINEAR

    def __call__(self, results):
        img = results['img']
        img_h, img_w = img.shape[:2]
        img = self.__cv2pil(img)
        img = img.convert('L')


        if img_w / img_h < 1.:
            # img = mmcv.imresize(img, (self.h, self.h))
            img = img.resize((self.h, self.h), self.interpolation)
            resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
            resize_img[0:self.h, 0:self.h, 0] = img
            img = resize_img
            width = self.h
        elif img_w / img_h < self.w / self.h:
            ratio = img_h / self.h
            new_w = int(img_w / ratio)
            # img = mmcv.imresize(img, (new_w, self.h))
            img = img.resize((new_w, self.h), self.interpolation)
            resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
            resize_img[0:self.h, 0:new_w, 0] = img
            img = resize_img
            width = new_w
        else:
            # img = mmcv.imresize(img, (self.w, self.h))
            img = img.resize((self.w, self.h), self.interpolation)
            resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
            resize_img[:, :, 0] = img
            img = resize_img
            width = self.w

        results['img'] = img
        results['img_shape'] = (self.h, self.w, 1)

        ratio = width / self.w
        # mask = results['mask']
        mask_h, mask_w = int(self.h / 8), int(self.w / 4)
        mask = np.zeros((1, mask_h, mask_w))
        mask_width = int(mask_w * ratio)
        mask[:, :, :mask_width] = 1
        results['mask'] = mask
        return results

    def __cv2pil(self, img):
        # tmp_img = img.copy()
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        img[:, :, 0] = r
        img[:, :, 1] = g
        img[:, :, 2] = b
        # img = img.astype(np.uint8)
        # img = Image.fromarray(img)
        return Image.fromarray(img.astype(np.uint8))

