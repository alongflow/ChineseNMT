# -*- coding: utf-8 -*-
# @Time    : 2023/12/20 23:07
# @Author  : liuyl
# @File    : tojson.py
# @Software: VSCode

import json

files = ['train', 'valid', 'test']
for file in files:
    sf = file if file != 'valid' else 'dev'
    save_path = './json/' + sf + '.json'
    zh_lines = []
    en_lines = []
    result = []
    with open('./en-zh/' + file + '.zh', 'r') as f:
        zh_lines.extend(f.readlines())
    with open('./en-zh/' + file + '.en', 'r') as f:
        en_lines.extend(f.readlines())
    for i in range(len(en_lines)):
        en_line = en_lines[i].strip()
        zh_line = zh_lines[i].strip()
        # 删除话外音
        zh_line = zh_line.replace('（鼓掌）', '').replace('（鼓掌声）', '').replace(
            '（众人鼓掌）',
            '').replace('（热烈鼓掌）',
                        '').replace('（观众鼓掌）',
                                    '').replace('（观众掌声）',
                                                '').replace('（鼓掌）', '')
        # 删除空样本
        if (not en_line) or (not zh_line):
            continue
        result.append([en_line, zh_line])
    with open(save_path, 'w') as f:
        json.dump(result, f)
