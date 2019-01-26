#-*- coding:utf-8 _*-
"""
@author:limuyu
@file: check_one.py
@time: 2019/01/26
@contact: limuyu0110@pku.edu.cn

检查一个模型目录里面的log
"""

import argparse
import codecs
import re

from os.path import join

# 用于解析log文档的正则表达式
# 'Dev: time: 59.71s, speed: 4.53st/s; acc: 0.9606, p: 0.7000, r: 0.5398, f: 0.6096'
re_dev = r'Dev: time: [0-9\.]+s, speed: [0-9\.]+st/s; acc: [\-\.0-9]+, p: ([\-\.0-9]+), r: ([\-\.0-9]+), f: ([\-\.0-9]+)'
# 'Epoch: 38/100'
re_epoch = r'Epoch: ([0-9]+)/[0-9]+'


def check(path):
    log_path = join(path, 'run.log')
    with codecs.open(log_path, 'r', 'utf8') as f:
        raw = f.read()
        devs = re.findall(re_dev, raw)
        # epochs = re.findall(re_epoch, raw)
        for i, d in enumerate(devs):
            print(F'Epoch:{i}, Dev:{d}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    args = parser.parse_args()

    check(args.path)
