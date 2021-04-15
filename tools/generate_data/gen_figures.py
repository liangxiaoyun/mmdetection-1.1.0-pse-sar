'''
_*_ coding: utf-8 _*_
@author: LiangXiaoyun
@time: 20210415
'''

import os
import cv2
import re
import pickle
import random
import click

import numpy as np

from glob import glob

import generate_chars
from get_figure import GetFigure


@click.command()
@click.option('--save_img_pth', default=None, help='Path to images to save.')
@click.option('--save_num', default=None, help='img number to save')
@click.option('--suffix', default='_1.jpg', help='the suffix added to figure name')#增加下标，方便后续直接追加数据

def main(save_img_pth, save_num, suffix):
    if not os.path.isdir(save_img_pth):
        os.mkdir(save_img_pth)

    # 读入字体库
    font_path = './font_ch'
    # 背景图库
    bg_root_path = 'bg_imgs/'
    bg_imgs_paths = glob(bg_root_path + "/" + r"*.jpg")

    font_sizes = list(range(15, 25))

    #通过读取txt文件来获取总的chars列表，来生成数据
    # chars_sequence = generate_chars.get_all_chars_in_txt('char_sequences.txt')
    chars_sequence = []

    # all alphabets  用于后续得字符删除/添加/替换得增强
    # alphabets = generate_chars.get_all_keys('alphabets.txt')
    alphabets = ''

    GF = GetFigure(bg_imgs_paths, font_sizes, font_path, chars_sequence, alphabets)

    labels_path = save_img_pth + '/labels.txt'
    gs = 0
    if os.path.exists(labels_path):  # 支持中断程序后，在生成的图片基础上继续
        with open(labels_path, 'r', encoding='utf-8') as f:
            lines = list(f.readlines())
            new_lines = [l for l in lines if suffix in l]
            print("lines:", new_lines)

        if len(new_lines) > 0:
            gs = int(new_lines[-1].strip().split('\t')[0].split('/')[-1].split('_')[0])
            print('Resume generating from step %d' % gs)

    f = open(labels_path, 'a', encoding='utf-8')
    print('start generating...')

    for i in range(gs + 1, int(save_num) + 1):
        try:
            gen_img, chars = GF.get_horizontal_text_picture()
            if gen_img.mode != 'RGB':
                gen_img = gen_img.convert('RGB')

            save_img_name = str(i).zfill(7) + suffix

            save_img_name = os.path.join(save_img_pth, save_img_name)
            gen_img.save(save_img_name)

            f.write(save_img_name + '\t' + chars + '\n')
            print('gennerating:-------' + save_img_name + 'chars:' + chars)
        except Exception as e:
            print(e)
    f.close()

if __name__ == '__main__':
    main()