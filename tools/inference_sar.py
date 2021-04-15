# coding: utf-8
import warnings
import os
import pandas as pd
import numpy as np
import torch

import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_recognizer

def init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    model = build_recognizer(config.model, test_cfg=config.test_cfg)
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

def inference_recognizer(model, img):
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
    mask_h, mask_w = cfg.data.test.pipeline[1]['img_scale']
    text_max_len = cfg.text_max_len + 2
    data = dict(img=img, target_variable=torch.zeros(text_max_len, 1, dtype=torch.long), mask=np.zeros((1, mask_h, mask_w)))
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device.index])[0]
    # forward the model
    with torch.no_grad():
        pre, score = model(return_loss=False, rescale=True, **data)
    return pre, score

def predict():
    config_file = 'configs/sar_r50.py'
    checkpoint_file = 'work_dirs/sar50/epoch_200.pth'

    # 100张自标注测试集
    img_folder = 'figure_folder'
    save_csv = 'mm_sar_out.csv'

    model = init_detector(config_file, checkpoint_file)
    value_list = []
    with open(os.path.join(img_folder, 'labels.txt'), 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            i = line[0]
            gt = line[-1]
            print(i)
            # if i != '991.jpg': continue
            img_pth = os.path.join(img_folder, i)

            flag = 'False'
            pre, score = inference_recognizer(model, img_pth)
            if gt == pre:
                flag = 'True'
            print('gt:{}, pre:{}, score:{}, flag:{}'.format(gt, pre, score, flag))
            value_list.append({'img_name': i, 'gt': gt, 'pre': pre, 'score': score, 'flag': flag})

    df_values = pd.DataFrame.from_dict(value_list)
    df_values.to_csv(save_csv, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    predict()