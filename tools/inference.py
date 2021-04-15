# coding: utf-8
import warnings
import os
import cv2
import six
from PIL import Image

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import wrap_fp16_model
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector
import matplotlib.pyplot as plt

def init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result, data['img_meta'][0][0]['scale_factor']

def predict():
    config_file = 'configs/psenet_r50.py'
    checkpoint_file = 'work_dirs/psenet_r50/epoch_100.pth'

    # 100张自标注测试集
    img_folder = '/61_data_lxy/data/price_sheet/table_pse_data/180_table_images'
    out_folder = '/61_data_lxy/data/price_sheet/output/mm_pse_output'

    model = init_detector(config_file, checkpoint_file)
    for i in os.listdir(img_folder):
        print(i)
        # if i != '991.jpg': continue
        img_pth = os.path.join(img_folder, i)
        org_img = cv2.imread(img_pth)
        if org_img is None: continue
        h, w = org_img.shape[:2]
        result, scale_factor = inference_detector(model, img_pth)
        preds, boxes_list = result

        if len(boxes_list):
            boxes_list = boxes_list / scale_factor
        cv2.drawContours(org_img, boxes_list.astype(int), -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(out_folder, i), org_img)

if __name__ == '__main__':
    predict()