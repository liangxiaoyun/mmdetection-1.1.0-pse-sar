import pickle
import random
import cv2
import numpy as np

class FontColor(object):
    def __init__(self, col_file):
        with open(col_file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            self.colorsRGB = u.load()
        self.ncol = self.colorsRGB.shape[0]
        self.colorsRGB = np.r_[self.colorsRGB[:, 0:3], self.colorsRGB[:, 6:9]].astype('uint8')
        self.colorsLAB = np.squeeze(cv2.cvtColor(self.colorsRGB[None, :, :], cv2.COLOR_RGB2Lab))

# 分析图片，获取最适宜的字体颜色
def get_bestcolor(color_lib, crop_lab):
    if crop_lab.size > 4800:
        crop_lab = cv2.resize(crop_lab, (100, 16))
    labs = np.reshape(np.asarray(crop_lab), (-1, 3))
    clf = KMeans(n_clusters=8)
    clf.fit(labs)
    total = [0] * 8
    for i in clf.labels_:
        total[i] = total[i] + 1
    clus_result = [[i, j] for i, j in zip(clf.cluster_centers_, total)]
    clus_result.sort(key=lambda x: x[1], reverse=True)
    color_sample = random.sample(range(color_lib.colorsLAB.shape[0]), 500)
    def caculate_distance(color_lab, clus_result):
        weight = [1, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05, 0.01]
        d = 0
        for c, w in zip(clus_result, weight):
            d = d + np.linalg.norm(c[0] - color_lab)
        return d
    color_dis = list(map(lambda x: [caculate_distance(color_lib.colorsLAB[x], clus_result), x], color_sample))
    color_dis.sort(key=lambda x: x[0], reverse=True)
    return tuple(color_lib.colorsRGB[color_dis[0][1]])

def get_fonts(font_path, font_sizes):
    fonts = {}
    font_files = os.listdir(font_path)
    for size in font_sizes:
        tmp = []
        for font_file in font_files:
            tmp.append((font_file, ImageFont.truetype(os.path.join(font_path, font_file), size)))
        fonts[size] = tmp
    return fonts

# 选择字体
def chose_font(fonts, font_sizes):
    f_size = random.choice(font_sizes)
    font = random.choice(fonts[f_size])
    # print(f_size)
    # index = random.randint(0, len(fonts[f_size])-1)
    # print(index)
    # font = fonts[f_size][index]
    return font