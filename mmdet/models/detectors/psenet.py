from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from mmdet.core import (bbox2result, bbox2roi, bbox_mapping, build_assigner,
                        build_sampler, merge_aug_bboxes, merge_aug_masks,
                        multiclass_nms)
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from .test_mixins import RPNTestMixin
from ..utils.pse import decode


class Muti_DiceLoss(nn.Module):
    def __init__(self, text_classify, n=6, ratio=3, Lambda=0.7, reduction='mean'):
        super(Muti_DiceLoss, self).__init__()
        self.n = n
        self.Lambda = Lambda
        self.ratio = ratio
        self.text_classify = text_classify
        self.reduction = reduction
    def forward(self, outputs, labels, training_mask):
        muti_class_loss = []
        text_loss = []
        kernel_loss = []
        for i in range(len(self.text_classify)):
            texts = outputs[:,(i+1)*self.n-1,:,:]
            kernels = outputs[:,i*self.n:(i+1)*self.n-1,:,:]
            gt_texts = labels[:,(i+1)*self.n-1,:,:]
            gt_kernels = labels[:,i*self.n:(i+1)*self.n-1,:,:]

            selected_mask = self.ohem_batch(texts, gt_texts, training_mask)
            selected_mask = selected_mask.to(outputs.device)

            loss_text = self.dice_loss(texts, gt_texts, selected_mask)

            loss_kernels = []

            mask0 = torch.sigmoid(texts).data.cpu().numpy()
            mask1 = training_mask.squeeze(1).data.cpu().numpy()
            selected_mask = ((mask0 > 0.5) & (mask1 > 0.5)).astype('float32')
            selected_mask = torch.from_numpy(selected_mask).float()
            selected_mask = selected_mask.to(outputs.device)
            kernels_num = gt_kernels.size()[1]
            for j in range(kernels_num):
                kernels_i = kernels[:, j, :, :]
                gt_kernels_i = gt_kernels[:, j, :, :]
                loss_kernel_i = self.dice_loss(kernels_i, gt_kernels_i, selected_mask)
                loss_kernels.append(loss_kernel_i)
            loss_kernels = torch.stack(loss_kernels).mean(0)
            if self.reduction == 'mean':
                loss_text = loss_text.mean()
                loss_kernels = loss_kernels.mean()
            elif self.reduction == 'sum':
                loss_text = loss_text.sum()
                loss_kernels = loss_kernels.sum()

            loss = self.Lambda * loss_text + (1-self.Lambda) * loss_kernels

            text_loss.append(loss_text)
            kernel_loss.append(loss_kernels)
            muti_class_loss.append(loss)

        muti_class_loss_avg = torch.stack(muti_class_loss).mean()
        text_loss_avg = torch.stack(text_loss).mean()
        kernel_loss_avg = torch.stack(kernel_loss).mean()

        return {'muti_class_loss_avg': muti_class_loss_avg, 'text_loss_avg': text_loss_avg, 'kernel_loss_avg':kernel_loss_avg}

    def dice_loss(self, pred, target, mask):
        pred = torch.sigmoid(pred)
        pred = pred.contiguous().view(pred.size()[0], -1)
        target = target.contiguous().view(target.size()[0], -1)
        mask = mask.contiguous().view(mask.size()[0], -1)

        pred = pred * mask
        target = target * mask
        a = torch.sum(pred*target, 1)
        b = torch.sum(pred*pred, 1) + 0.0001
        c = torch.sum(target*target, 1) + 0.0001
        d = (2*a) / (b+c)
        return 1-d

    def ohem_batch(self, scores, gt_texts, training_mask):
        scores = scores.data.cpu().numpy()
        gt_texts = gt_texts.data.cpu().numpy()
        training_mask = training_mask.data.cpu().numpy()

        selected_masks = []
        for i in range(scores.shape[0]):
            selected_masks.append(self.ohem_single(scores[i,:,:], gt_texts[i, :,:], training_mask[i, :, :]))
        selected_masks = np.concatenate(selected_masks, 0)
        selected_masks = torch.from_numpy(selected_masks).float()
        return selected_masks

    def ohem_single(self,score, gt_text, training_mask):
        pos_num = (int)(np.sum(gt_text > 0.5)) - (int)(np.sum((gt_text > 0.5) & (training_mask <= 0.5)))
        if pos_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')
            return selected_mask
        neg_num = (int)(np.sum(gt_text <= 0.5))
        neg_num = (int)(min(pos_num*self.ratio, neg_num))

        if neg_num == 0:
            selected_mask = training_mask
            selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')
            return selected_mask

        neg_score = score[gt_text <= 0.5]
        neg_score_sorted = np.sort(-neg_score)
        threshold = -neg_score_sorted[neg_num-1]
        selected_mask = ((score >= threshold) | (gt_text > 0.5)) & (training_mask > 0.5)
        selected_mask = selected_mask.reshape(1, selected_mask.shape[1], selected_mask.shape[2]).astype('float32')
        return selected_mask

@DETECTORS.register_module
class PSENet(BaseDetector, RPNTestMixin):

    def __init__(self,
                 backbone,
                 neck=None,
                 text_classify=['text'],
                 n=6,
                 Lambda=0.7,
                 scale=1,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None):
        super(PSENet, self).__init__()

        self.scale = scale
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.init_weights(pretrained=pretrained)

        self.criterion = Muti_DiceLoss(text_classify=text_classify, n=n, Lambda=Lambda, reduction='mean')
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self, pretrained=None):
        super(PSENet, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      score_maps,
                      training_mask,
                      gt_bboxes_ignore=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)
        img_size = np.array(img.size()[2:])
        if self.with_neck:
            x = self.neck(x, img_size)

        loss = self.criterion(x, score_maps, training_mask)

        return loss

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Run inference on a single image.

        Args:
            img (Tensor): must be in shape (N, C, H, W)
            img_meta (list[dict]): a list with one dictionary element.
                See `mmdet/datasets/pipelines/formatting.py:Collect` for
                details of meta dicts.
            proposals : if specified overrides rpn proposals
            rescale (bool): if True returns boxes in original image space

        Returns:
            dict: results
        """
        x = self.extract_feat(img)
        img_size = np.array(img.size()[2:])
        if self.with_neck:
            x = self.neck(x, img_size)
        preds = x

        pred, boxes_list = decode(preds[0][:6], self.scale)

        return pred, boxes_list


    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        x = self.extract_feat(imgs)
        img_size = np.array(imgs.size()[2:])
        if self.with_neck:
            x = self.neck(x, img_size)
        preds = x
        result = []
        for x, img_meta in zip(preds, img_metas):
            pred, booxes_list = decode(x[0][:6], self.scale)
            result.append(booxes_list)

        return result


    def show_result(self, data, result, **kwargs):
        if self.with_mask:
            ms_bbox_result, ms_segm_result = result
            if isinstance(ms_bbox_result, dict):
                result = (ms_bbox_result['ensemble'],
                          ms_segm_result['ensemble'])
        else:
            if isinstance(result, dict):
                result = result['ensemble']
        super(PSENet, self).show_result(data, result, **kwargs)
