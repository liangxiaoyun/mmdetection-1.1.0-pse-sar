from __future__ import division

import numpy as np
import torch
import random
import cv2
import torch.nn as nn
import torch.nn.functional as F

from .. import builder
from ..registry import RECOGNIZER
from .base import BaseRecognizer

from .encoder import Encoder
from .decoder import Decoder
from .sar_resnet import SAR_ResNet, BasicBlock
from .str_lable_converter_for_attention import strLabelConverterForAttention

@RECOGNIZER.register_module
class SAR(BaseRecognizer):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 nh=512,
                 nclass=37,
                 dropout_p=0.1,
                 text_max_len=None,#100
                 teacher_forcing_ratio=1.0,
                 alphabet=None,
                 noy=False,
                 show_attention=False,
                 training=True,
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None):
        super(SAR, self).__init__()

        self.backbone = SAR_ResNet(BasicBlock, [1, 2, 5, 3])
        if neck is not None:
            self.neck = builder.build_neck(neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.nh = nh
        self.nclass = nclass
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.show_attention = False if training else show_attention

        self.noy = noy
        if text_max_len is not None:
            self.position_embedding = nn.Embedding(text_max_len + 2, self.nh)

        self.Encoder = Encoder(nh)
        self.Decoder = Decoder(nh, nclass, dropout_p, noy=self.noy, show_attention=self.show_attention)
        self.converter = strLabelConverterForAttention(alphabet, text_max_len)

    def extract_feat(self, img):
        x = self.backbone(img)
        return x

    def forward_train(self,
                      img,
                      img_meta,
                      target_variable,
                      target_cp,
                      mask,
                      gt_bboxes_ignore=None,
                      proposals=None):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                torch.Size([128, 1, 48, 256])

            target_variable:torch.Size([128, 1, 1, 27])

            target_cp:torch.Size([128, 1, 1, 27])

            mask:torch.Size([128, 1, 6, 64])

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #layernorm
        # img = F.layer_norm(img, img.size()[1:])
        feature = self.extract_feat(img)
        hidden = self.Encoder(feature)
        hidden = hidden.permute(2, 0, 1)

        target_variable = target_variable.squeeze(1).squeeze(1)
        target_cp = target_cp.squeeze(1).squeeze(1)

        decoder_input = target_variable[:, 0].cuda()
        decoder_outputs = []
        score = 1.0

        for di in range(1, target_variable.shape[1]):
            ###############
            #为了减少上一状态对下一状态解码的影响，将y输入改为位置信息输入
            if self.noy:
                di = torch.full([target_variable.shape[0]], di).long().cuda()
                decoder_input = self.position_embedding(di)
            ###############
            decoder_output, hidden = self.Decoder(decoder_input, hidden, feature, mask)

            decoder_outputs.append(decoder_output.unsqueeze(1))
            teacher_force = random.random() < self.teacher_forcing_ratio

            if teacher_force:
                decoder_input = target_variable[:, di]
            else:
                topv, topi = decoder_output.data.topk(1)
                score *= topv[0, 0].item()
                decoder_input = topi.squeeze(1)


        decoder_outputs = torch.cat(decoder_outputs, 1)

        loss = F.cross_entropy(decoder_outputs.contiguous().view(-1, self.nclass), target_cp[:, 1:].contiguous().view(-1), ignore_index=-1)

        return {'loss':loss}

    def simple_test(self, img, img_meta, target_variable, mask, proposals=None, rescale=False):
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
        feature = self.extract_feat(img)
        hidden = self.Encoder(feature)
        hidden = hidden.permute(2, 0, 1)

        target_variable = target_variable[0].squeeze(1).squeeze(1)
        mask = mask[0]

        decoder_input = target_variable[:, 0].cuda()
        decoder_outputs = []
        w_outputs, hidden_outputs = [], []
        score = 1.0
        if self.show_attention:
            #img = img * 0.5 + 0.5
            ori_img = img.cpu().squeeze(0).numpy().transpose(1,2,0) * 255

        for di in range(1, target_variable.shape[1]):
            if self.noy:
                di = torch.full([target_variable.shape[0]], di).long().cuda()
                decoder_input = self.position_embedding(di)

            if self.show_attention:
                decoder_output, hidden, w = self.Decoder(decoder_input, hidden, feature, mask)
                w_outputs.append(w)
                hidden_outputs.append(hidden[1:].permute(1,2,0).unsqueeze(3))
            else:
                decoder_output, hidden = self.Decoder(decoder_input, hidden, feature, mask)

            decoder_output = F.softmax(decoder_output, dim=1)
            decoder_outputs.append(decoder_output.unsqueeze(1))

            topv, topi = decoder_output.data.topk(1)
            score *= topv[0, 0].item()
            decoder_input = topi.squeeze(1)
            if topi.squeeze(1) == 0:
                break

        decoder_outputs = torch.cat(decoder_outputs, 1)
        s_score = np.power(score, 1.0 / decoder_outputs.shape[1])

        topv, topi = decoder_outputs.data.topk(1)
        ni = topi.squeeze(2)
        predict_labels = []
        for x in range(ni.shape[0]):
            pred = ni[x]
            predict_label = ''
            for i in range(len(pred)):
                if pred[i] == 0:
                    break
                l = self.converter.decode(pred[i])
                predict_label += l

                if self.show_attention:
                    weight = w_outputs[i].cpu().numpy().reshape((6,64)) * 255
                    weight = cv2.resize(weight, (256, 48))[:,:,None]
                    weight_img = 0.5 * ori_img + 0.5 * weight
                    if i > 0:
                        diff_hidden = torch.norm(hidden_outputs[i] - hidden_outputs[i-1])
                    else:
                        diff_hidden = 0
                    cv2.imwrite('{}_{}_{}.jpg'.format(i, l, diff_hidden), weight_img.astype('uint8'))

            predict_labels.append(predict_label)

        return predict_labels[0], s_score


    def aug_test(self, imgs, img_metas, proposals=None, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        pass


    def show_result(self, data, result, **kwargs):
        pass
