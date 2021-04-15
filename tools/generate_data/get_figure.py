import random
import cv2

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from sklearn.cluster import KMeans

from fontcolor import FontColor, get_fonts, chose_font, get_bestcolor
import generate_chars

class GetFigure(object):
    def __init__(self, bg_imgs_paths, font_sizes, font_path, chars_sequence=[], alphabets=''):
        # 读入字体色彩库
        self.color_lib = FontColor('colors_new.cp')
        self.fonts = get_fonts(font_path, font_sizes)
        self.font_sizes = font_sizes
        self.bg_imgs_paths = bg_imgs_paths
        self.chars_sequence = chars_sequence
        self.alphabets = alphabets

    def rotate_img_box(self, image, box, angle):
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        rotate_angle = angle
        M = cv2.getRotationMatrix2D((cX, cY), rotate_angle, 1)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotate_image = cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        rotate_box = []

        for ponit in box:
            rot_point = np.dot(M, np.array([ponit[0], ponit[1], 1]))
            rotate_box.append([np.int(rot_point[0]), np.int(rot_point[1])])

        new_box = [min(rotate_box[0][0], rotate_box[1][0], rotate_box[2][0], rotate_box[3][0]),
                   min(rotate_box[0][1], rotate_box[1][1], rotate_box[2][1], rotate_box[3][1]),
                   max(rotate_box[0][0], rotate_box[1][0], rotate_box[2][0], rotate_box[3][0]),
                   max(rotate_box[0][1], rotate_box[1][1], rotate_box[2][1], rotate_box[3][1])]

        return Image.fromarray(rotate_image.astype('uint8')).convert('RGB'), new_box

    def bg_img_refine(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        w, h = img.size
        if w < 400:
            img = img.resize((400, h), Image.ANTIALIAS)
            w = 400
        if h < 30:
            img = img.resize((w, 30), Image.ANTIALIAS)
            h = 30
        return img, w, h

    def get_horizontal_text_picture(self, char_process_p=0.0, big_distance_p=0.0):
        bg_img_path = random.choice(self.bg_imgs_paths)
        retry = 0
        img = Image.open(bg_img_path)
        img, w, h = self.bg_img_refine(img)
        x1 = 0  # text的开始位置
        y1 = 0

        #找到合适的字符串、crop位置和文字颜色
        while True:
            chars = generate_chars.gen_vin()
            if random.random() < char_process_p:#字符串处理
                chars = generate_chars.character_process(chars, alphabets)

            if (chars is None or chars == ''):
                continue

            font_name, font = chose_font(self.fonts, self.font_sizes)
            print('font_name: ', font_name)

            f_w, f_h = font.getsize(chars)
            if f_w < w:
                if (w - f_w) < 1:
                    print("if (w - f_w)<1:")
                    continue
                if (h - f_h) < 1:
                    print("if (h - f_h)<1:")
                    continue
                x1 = random.randint(0, w - f_w - 1)
                y1 = random.randint(0, h - f_h - 1)
                x2 = x1 + f_w
                y2 = y1 + f_h

                # 随机加一点偏移
                rd = random.random()
                if rd < 0.8:  # 设定偏移的概率
                    crop_x1 = int(max(0, x1 - random.uniform(0, f_h / 3)))
                    crop_x2 = int(min(w - 1, x2 + random.uniform(0, f_h / 3)))
                    crop_y1 = int(max(0, y1 - random.uniform(0, f_h / 6)))
                    crop_y2 = int(min(h - 1, y2 + random.uniform(0, f_h / 6)))
                else:
                    crop_y1 = y1
                    crop_x1 = x1
                    crop_y2 = y2
                    crop_x2 = x2
                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)
                if np.linalg.norm(
                        np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 35 and retry < 30:  # 颜色标准差阈值，颜色太丰富就不要了
                    retry = retry + 1
                    print("retry = retry+1")
                    print('bg_image_file:   ', image_file)
                    continue
                best_color = get_bestcolor(color_lib, crop_lab)
                break
            else:
                print("pass:")
                pass

        draw = ImageDraw.Draw(img)

        #单字符写，扩大字符间距离
        if random.random() < big_distance_p:
            c_w = 0
            interval = random.randint(2, 6)
            # print(interval)
            for nn, c in enumerate(chars):
                draw.text((x1 + c_w, y1), c, fill=best_color, font=font)
                c_w += font.getsize(c)[0] + interval

            crop_x2 = int(x1 + c_w)
            # draw.text((crop_x2 + 1, y1), chars, fill=best_color, font=font)
            # crop_x2 = crop_x2 + font.getsize(chars)[0]
            if crop_x2 > w - 1:
                img = ori_img.resize((crop_x2, h), Image.ANTIALIAS)
                draw = ImageDraw.Draw(img)
                c_w = 0
                for nn, c in enumerate(chars):
                    draw.text((x1 + c_w, y1), c, fill=best_color, font=font)
                    c_w += font.getsize(c)[0] + interval

        #使用正常的font间距
        else:
            draw.text((x1, y1), chars, fill=best_color, font=font)

        # 画水平直线
        if random.random() < 0.5:
            start_x = random.randint(crop_x1, crop_x2 - 1)
            end_x = random.randint(min(start_x + 15, crop_x2 - 1), crop_x2)
            start_y = random.choice([random.randint(crop_y1, min(crop_y1 + 20, crop_y2)), random.randint(min(crop_y1, crop_y2 - 20), crop_y2)])
            end_y = min(start_y + random.randint(1,3), crop_y2)
            draw.rectangle((start_x, start_y, end_x, end_y), best_color)

        # 旋转
        if random.random() < 0.3:
            angle = random.randint(-5, 5)
            # crop_img = crop_img.rotate(random.randint(-5,5))
            # img.resize((crop_x2-crop_x1, crop_y2- crop_y1),Image.ANTIALIAS)
            rotate_img, rotate_box = self.rotate_img_box(np.array(img),
                                                    [[crop_x1, crop_y1], [crop_x1, crop_y2], [crop_x2, crop_y2],
                                                     [crop_x2, crop_y1]], angle)
            crop_img = rotate_img.crop((rotate_box[0], rotate_box[1], rotate_box[2], rotate_box[3]))
        else:
            crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

        # 加模糊
        if random.random() < 0.4:
            mohu = random.randint(1, 3)
            crop_img = Image.fromarray(cv2.blur(np.array(crop_img), (mohu, mohu)).astype('uint8')).convert('RGB')
            # crop_img = crop_img.filter(ImageFilter.GaussianBlur)

        return crop_img, chars

    def get_vertical_text_picture(self, char_process_p=0.0):
        bg_img_path = random.choice(self.bg_imgs_paths)
        img = Image.open(bg_img_path)
        img, w, h = self.bg_img_refine(img)
        retry = 0
        x1 = 0  # text的开始位置
        y1 = 0

        while True:
            chars = generate_chars.gen_vin()
            if random.random() < char_process_p:  # 字符串处理
                chars = generate_chars.character_process(chars, alphabets)

            if (chars is None or chars == ''):
                continue

            font_name, font = chose_font(self.fonts, self.font_sizes)
            print('font_name: ', font_name)

            ch_w = []
            ch_h = []
            for ch in chars:
                wt, ht = font.getsize(ch)
                ch_w.append(wt)
                ch_h.append(ht)
            f_w = max(ch_w)
            f_h = sum(ch_h)

            # 完美分割时应该取的,也即文本位置
            if h > f_h:
                x1 = random.randint(0, w - f_w)
                y1 = random.randint(0, h - f_h)
                x2 = x1 + f_w
                y2 = y1 + f_h
                # 随机加一点偏移
                rd = random.random()
                if rd < 0.2:  # 设定偏移的概率
                    crop_x1 = x1 - random.random() / 4 * f_w
                    crop_y1 = y1 - random.random() / 2 * f_w
                    crop_x2 = x2 + random.random() / 4 * f_w
                    crop_y2 = y2 + random.random() / 2 * f_w
                    crop_y1 = int(max(0, crop_y1))
                    crop_x1 = int(max(0, crop_x1))
                    crop_y2 = int(min(h, crop_y2))
                    crop_x2 = int(min(w, crop_x2))
                else:
                    crop_y1 = y1
                    crop_x1 = x1
                    crop_y2 = y2
                    crop_x2 = x2

                crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                crop_lab = cv2.cvtColor(np.asarray(crop_img), cv2.COLOR_RGB2Lab)
                if np.linalg.norm(
                        np.reshape(np.asarray(crop_lab), (-1, 3)).std(axis=0)) > 35 and retry < 30:  # 颜色标准差阈值，颜色太丰富就不要了
                    retry = retry + 1
                    continue
                best_color = get_bestcolor(color_lib, crop_lab)
                break

            else:
                pass
        draw = ImageDraw.Draw(img)
        i = 0

        for ch in chars:
            draw.text((x1, y1), ch, best_color, font=font)
            y1 = y1 + ch_h[i]
            i = i + 1

        crop_img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        crop_img = crop_img.transpose(Image.ROTATE_270)
        return crop_img, chars

