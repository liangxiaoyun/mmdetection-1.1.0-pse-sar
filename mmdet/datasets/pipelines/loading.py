import os.path as osp

import cv2
import pyclipper
import mmcv
import numpy as np
import pycocotools.mask as maskUtils

from ..registry import PIPELINES

@PIPELINES.register_module
class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = osp.join(results['img_prefix'],
                                results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']
        img = mmcv.imread(filename, self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

    def __repr__(self):
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['bboxes']

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str

@PIPELINES.register_module
class LoadAnnotations_PSE(object):

    def __init__(self,
                 n=6,
                 m=0.5,
                 result_num=1,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_poly_mask=False,
                 poly2mask=False):
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.poly2mask = poly2mask
        self.with_poly_mask = with_poly_mask
        self.n = n
        self.m = m
        self.result_num = result_num

    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        bboxes = ann_info['bboxes']
        new_bboxes = []
        for b in bboxes:
            new_bboxes.append([[b[0],b[1]], [b[2],b[1]], [b[2], b[3]], [b[0], b[3]]])
        results['gt_bboxes'] = np.array(new_bboxes, dtype=np.float32)

        gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bbox_fields'].append('gt_bboxes')
        return results

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results

    def _generate_rbox(self, im_size, result_num, text_polys, text_class, i, n, m):
        """
        生成mask图，白色部分是文本，黑色是北京
        :param im_size: 图像的h,w
        :param text_polys: 框的坐标
        :param text_tags: 标注文本框是否参与训练
        :return: 生成的mask图
        """
        h, w = im_size
        score_map = np.zeros((h, w, result_num), dtype=np.uint8)
        score_map_text1 = score_map.copy()
        score_map_text2 = score_map.copy()

        for poly, class_text in zip(text_polys, text_class):
            poly = poly.astype(np.int)
            r_i = 1 - (1 - m) * (n - i) / (n - 1)
            if cv2.arcLength(poly, True) == 0:
                continue
            d_i = cv2.contourArea(poly) * (1 - r_i * r_i) / cv2.arcLength(poly, True)
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(-d_i))

            if class_text == 0:
                cv2.fillPoly(score_map_text1, [shrinked_poly], (1,0))

            elif class_text == 1:
                cv2.fillPoly(score_map_text2, [shrinked_poly], (0, 1))

        score_map += score_map_text1
        score_map += score_map_text2

        return score_map

    def _get_kernel_mask(self, results):
        im_size = results['img_shape'][:2]
        ann_info = results['ann_info']
        bboxes = ann_info['bboxes']
        new_bboxes = []
        for b in bboxes:
            new_bboxes.append([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]])
        text_polys = np.array(new_bboxes, dtype=np.float32)

        score_maps = []
        training_mask = np.ones(im_size, dtype=np.uint8)
        text_class = [i-1 for i in results['ann_info']['labels']]   #每个box的类别
        for i in range(1, self.n+1):
            score_map = self._generate_rbox(im_size, self.result_num, text_polys, text_class, i, self.n, self.m)
            score_maps.append(score_map)
        # score_maps = np.array(np.concatenate(score_maps, axis=-1), dtype=np.float32)
        results['score_maps'] = score_maps
        results['training_mask'] = training_mask
        return results

    def _load_semantic_seg(self, results):
        results['gt_semantic_seg'] = mmcv.imread(
            osp.join(results['seg_prefix'], results['ann_info']['seg_map']),
            flag='unchanged').squeeze()
        results['seg_fields'].append('gt_semantic_seg')
        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        if self.with_poly_mask:
            results = self._get_kernel_mask(results)
        if self.with_seg:
            results = self._load_semantic_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={},'
                     ' with_seg={})').format(self.with_bbox, self.with_label,
                                             self.with_mask, self.with_seg)
        return repr_str

@PIPELINES.register_module
class LoadAnnotations_SAR(object):

    def __init__(self):
        pass

    def _get_label(self, results):
        im_size = results['img_shape'][:2]
        label = results['img_info']['label']

        mask_w, mask_h = int(im_size[1]/4), int(im_size[0]/8)
        mask = np.zeros((1, mask_h, mask_w))

        results['target_variable'] = label
        results['target_cp'] = label
        results['mask'] = mask
        return results

    def __call__(self, results):
        results = self._get_label(results)
        return results

@PIPELINES.register_module
class LoadProposals(object):

    def __init__(self, num_max_proposals=None):
        self.num_max_proposals = num_max_proposals

    def __call__(self, results):
        proposals = results['proposals']
        if proposals.shape[1] not in (4, 5):
            raise AssertionError(
                'proposals should have shapes (n, 4) or (n, 5), '
                'but found {}'.format(proposals.shape))
        proposals = proposals[:, :4]

        if self.num_max_proposals is not None:
            proposals = proposals[:self.num_max_proposals]

        if len(proposals) == 0:
            proposals = np.array([[0, 0, 0, 0]], dtype=np.float32)
        results['proposals'] = proposals
        results['bbox_fields'].append('proposals')
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(num_max_proposals={})'.format(
            self.num_max_proposals)
