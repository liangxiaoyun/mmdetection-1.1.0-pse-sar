# -*- coding: utf-8 -*-
import os
import shutil
import json
import cv2
import numpy as np

classname2id = {'tampered':1} #0是背景

class txt2coco:
    def __init__(self, image_dir, mask_file):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.image_dir = image_dir
        self.total_annos = {}
        self.mask_file = mask_file

    # 构建类别
    def _init_categories(self):
        for k, v in classname2id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # 构建COCO的image字段
    def _image(self, path):
        image = {}
        print(path)
        img = cv2.imread(os.path.join(self.image_dir, path))
        image['height'] = img.shape[0]
        image['width'] = img.shape[1]
        image['id'] = self.img_id
        image['file_name'] = path
        return image

    # 构建COCO的annotation字段
    def _annotation(self, points, label=None):
        label = 'tampered' if label is None else label
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname2id[label])
        annotation['segmentation'] = self._get_seg(points)
        box, area = self._get_box_area(points)
        annotation['bbox'] = box
        annotation['iscrowd'] = 0
        annotation['area'] = area
        return annotation

    # COCO的格式： [x1,y1,w,h] 对应COCO的bbox格式
    def _get_box_area(self, points):
        #points [x1,y1,x2,y2,....]
        x, y, w, h = cv2.boundingRect(np.array(points).reshape(-1, 2).astype(int))  # x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
        area = w * h
        return [x, y, w, h], area

    # segmentation
    def _get_seg(self, points):
        return [points]

    def savejson(self, instance, save_pth):
        json.dump(instance, open(save_pth, 'w'), ensure_ascii=False, indent=2)

    # 由txt文件构建COCO
    def to_coco(self):
        self._init_categories()
        for key in os.listdir(self.image_dir):
            self.images.append(self._image(key))
            annos = self.total_annos[key.split('.')[0]]
            for point in annos:
                annotation = self._annotation(point)
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    def parse_mask_img(self):
        for mask_file in os.listdir(self.mask_file):
            self.total_annos[mask_file.split('.')[0]] = []
            gray = cv2.imread(os.path.join(self.mask_file, mask_file), 0)
            # Grey2Binary
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            # for i in range(4):
            #     binary = cv2.dilate(binary, kernel)
            # 轮廓检测
            contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                points = []
                # for p in contour:
                #     points.extend([float(p[0][0]), float(p[0][1])])
                rect = cv2.minAreaRect(contour)  # 得到最小外接矩形的（中心(x,y), (宽,高), 旋转角度）
                box = cv2.boxPoints(rect)  # 获取最小外接矩形的4个顶点坐标
                for p in box:
                    points.extend([float(p[0]), float(p[1])])
                self.total_annos[mask_file.split('.')[0]].append(points)

    def parse_txt(self):
        for txt_file in os.listdir(self.gt):
            self.total_annos[txt_file.split('.')[0]] = []
            with open(os.path.join(self.gt, txt_file), 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    points = line.split('\t')[:8]
                    points = list(map(float, points))
                    self.total_annos[txt_file.split('.')[0]].append(points)

def main(mask_file, imgfile, save_pth):
    func = txt2coco(imgfile, mask_file)
    func.parse_mask_img()
    instance = func.to_coco()
    func.savejson(instance, save_pth)

if __name__ == '__main__':
    mask_file = '/Users/duoduo/Desktop/天池图片篡改检测/s2_data/data/small_train/mask'#'/liangxiaoyun583/data/FakeImg_Detection_Competition/data/tianchi_1_data/crop_mask_polygon'
    imgfile = '/Users/duoduo/Desktop/天池图片篡改检测/s2_data/data/small_train/images'#'/liangxiaoyun583/data/FakeImg_Detection_Competition/data/tianchi_1_data/crop_images'
    save_pth = '/Users/duoduo/Desktop/天池图片篡改检测/s2_data/data/small_train/train.json'

    func = txt2coco(imgfile, mask_file)
    # func.parse_mask_img()
    func.parse_txt()
    instance = func.to_coco()
    func.savejson(instance, save_pth)