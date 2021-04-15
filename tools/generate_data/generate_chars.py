# -*- coding: utf-8 -*-
import os
import random
import re
from faker import Faker
import del_alphabet
faker = Faker("zh-cn")

# 从列表中随机获取chars
def get_chars(char_sequences):
    while True:
        char_line = random.choice(char_sequences)
        if len(char_line) > 0:
            break
    line_len = len(char_line)
    char_len = random.randint(1, 20)  # 最多20个字

    if line_len <= char_len:
        return char_line
    chars = ''
    for i in range(char_len):
        idex = random.randint(0, line_len - 1)
        chars = chars + (char_lines[0][idex])
    '''  
    char_start = random.randint(0,line_len-char_len)
    chars = char_line[char_start:(char_start+char_len)]
    '''
    # print("chars:",chars)
    return chars

def separate_chinese_englishdigital(chars):
    chinese = re.sub("[A-Za-z0-9]", "", chars)
    englishdigital = ''.join(re.findall(r'[A-Za-z0-9]', chars))
    return chinese, englishdigital

def random_del_char(character):
    '''
    字符长度》3，则随机删除1~长度/2个字符(<6)，否则直接返回
    '''
    n = len(character)
    if n < 3:
        return character
    del_n = min(random.randint(1, int(n / 2)), 3)
    del_list = random.sample([i for i in range(n)], del_n)
    new_character = ''
    for i, c in enumerate(character):
        if i in del_list:
            continue
        else:
            new_character += c
    return new_character

def random_replace_char(character, keys):
    '''
    字符长度》3，则随机替换1~长度/2个字符(<6)，否则直接返回
    '''
    n = len(character)
    if n < 3:
        return character
    re_n = min(random.randint(1, int(n / 2)), 3)
    re_list = random.sample([i for i in range(n)], re_n)
    new_character = ''
    for i, c in enumerate(character):
        if i in re_list:
            new_character += random.choice(keys)
        else:
            new_character += c
    return new_character

def random_add_char(character, keys):
    '''
    随机增加1~长度/2个字符(<6)
    '''
    new_character = ''
    n = len(character)
    add_n = min(random.randint(1, max(int(n / 2), 2)), 3)
    if n < 3:
        new_character = character
        for i in range(add_n):
            new_character += random.choice(keys)

    else:
        add_list = random.sample([i for i in range(n)], add_n)
        for i, c in enumerate(character):
            if i in add_list:
                new_character += random.choice(keys)
            new_character += c
    return new_character

#随机删除，增加，替换
def character_process(character, keys):
    tmp = random.random()
    if tmp < 0.3:
        character = random_del_char(character)
    elif tmp < 0.6:
        character = random_replace_char(character, keys)
    else:
        character = random_add_char(character, keys)
    return character

def load_alphabet(alphabet_pth):
    alphabet = ''
    with open(alphabet_pth, 'r', encoding='utf-8') as total_file:
        total_txt = total_file.readlines()
        for line_txt in total_txt:
            alphabet = alphabet + line_txt
    alphabet = alphabet.replace('\n', '').replace(' ', '') + ' '
    alphabet = list(alphabet)
    return alphabet

def get_all_keys(alphabets_pth):
    al = ''
    alphabet = ''
    file = os.path.splitext(alphabets_pth)
    filename, type = file
    raise type == '.txt'
    with open(alphabets_pth, 'r', encoding='utf-8') as f:
        total_txt = f.readlines()
        for line_txt in total_txt:
            al += line_txt
        alphabet = al.replace('\n', '').replace(' ', '') + ' '

    return alphabet

def get_all_chars_in_txt(pth='color.txt'):
    se = []
    with open(pth, 'r') as f:  # , encoding='gbk'
        for l in f.readlines():
            se.append(l.strip())
    return se

def gen_letter(char_length=[1,3]):
    num = random.randint(char_length[0], char_length[1])
    target = ''
    for i in range(num):
        if i != 0:
            target += ' '
        target += ''.join(random.sample('ABCDEFGHIJKLMNOPQRSTUVWXYZ', random.randint(3, 8)))
    return target

def generate_fake_name():
    fa = faker.simple_profile(sex=None)
    name = fa.get('name')
    return name

def gen_vin():
    gen_vin_value = ''
    len_vin_value_1 = 13
    len_vin_value_2 = 4
    PART_1 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
              'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    PART_2 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    for i in range(len_vin_value_1):
        gen_vin_value = gen_vin_value + (random.choice(PART_1))
    for j in range(len_vin_value_2):
        gen_vin_value = gen_vin_value + (random.choice(PART_2))

    return gen_vin_value