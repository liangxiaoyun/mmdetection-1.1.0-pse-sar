import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import caffe2_xavier_init

from ..registry import NECKS
from ..utils import ConvModule
import torch

@NECKS.register_module
class PSEFPN(nn.Module):
    """NAS-FPN.

    NAS-FPN: Learning Scalable Feature Pyramid Architecture for Object
    Detection. (https://arxiv.org/abs/1904.07392)
    """

    def __init__(self,
                 result_num=6,
                 scale=1):
        super(PSEFPN, self).__init__()
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        self.conv = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(inplace=True))
        self.out_conv = nn.Conv2d(256, result_num, kernel_size=1, stride=1)
        self.scale = scale

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:], mode='bilinear', align_corners=True) + y

    def _upsample_cat(self, p2,p3,p4,p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=True)
        p5 = F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=True)
        return torch.cat([p2, p3, p4, p5], dim=1)

    def forward(self, inputs, im_size):
        H, W = im_size
        #img:(8,3,640,640)
        #c2:(8,256,160,160) c3:(8,512,80,80),c4:(8,1024,40,40),c5:(8,2048,20,20)
        c2,c3,c4,c5 = inputs
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p2 = self._upsample_add(p3, self.latlayer3(c2))

        p4 = self.smooth1(p4)
        p3 = self.smooth1(p3)
        p2 = self.smooth1(p2)

        x = self._upsample_cat(p2,p3,p4,p5)
        x = self.conv(x)
        x = self.out_conv(x)
        x = F.interpolate(x, size=(H // self.scale, W // self.scale), mode='bilinear', align_corners=True)
        return x


