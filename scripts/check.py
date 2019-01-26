#-*- coding:utf-8 _*-
"""
@author:limuyu
@file: check.py
@time: 2019/01/26
@contact: limuyu0110@pku.edu.cn

和多进程程序tiao.py 配合，指定一个根，按深度优先遍历文档树
实时把dev上的结果反映到一个jsonl文档中，配合linux的tail命令监控训练进度
"""

import codecs
import os
from os.path import join, isfile
import argparse
import json
import re
from collections import defaultdict
from time import sleep

# 这个目录是nfs上面默认的
path = '/data/nfsdata/nlp/projects'
# 就是tiao.py中的src_folder 即树根目录
root_folder = 'MUYUWEIBO'
# 暂定想要把更新的状态写入一个json文件，这里需要指明其路径
json_file = 'tmp_res.txt'

# 用于解析log文档的正则表达式
# 'Dev: time: 59.71s, speed: 4.53st/s; acc: 0.9606, p: 0.7000, r: 0.5398, f: 0.6096'
re_dev = r'Dev: time: [0-9\.]+s, speed: [0-9\.]+st/s; acc: [\-\.0-9]+, p: ([\-\.0-9]+), r: ([\-\.0-9]+), f: ([\-\.0-9]+)'
# 'Epoch: 38/100'
re_epoch = r'Epoch: ([0-9]+)/[0-9]+'


# 递归函数
def recur(sub_tree_root, status_dict):
    """

    :param sub_tree_root:
        这应该是一个绝对路径
    :return:
    """
    sons = os.listdir(sub_tree_root)

    # 如果已经到叶节点，开始解析run.log的结果
    if 'run.log' in sons:
        with codecs.open(join(sub_tree_root, 'run.log'), 'r', 'utf8') as f:
            raw = f.read()
            params = re.split(r'[\/\+]', sub_tree_root)
            # print(params)
            PID = params[-2]
            # print(PID)
            # input()
            param_hash = '\n'.join(sub_tree_root.split('/')[5:-2])
            # print(param_hash)
            # input()
            try:
                newest_dev = [float(d) for d in re.findall(re_dev, raw)[-1]]
            except Exception:
                return
            epoch = int(re.findall(re_epoch, raw)[-1])
            if status_dict[param_hash]['epoch'] < int(epoch):
                status_dict[param_hash]['epoch'] = epoch
                status_dict[param_hash]['newest_dev'] = newest_dev
                if newest_dev[-1] > status_dict[param_hash]['best_dev'][-1]:
                    status_dict[param_hash]['best_dev'] = newest_dev
                with codecs.open(json_file, 'a', 'utf8') as fw:
                    fw.write('=' * 70 + '\n')
                    fw.write(F'MODEL_NAME:\n\t{param_hash}\n{"-"*50}\n')
                    fw.write(F'MODEL_PATH\n')
                    fw.write(sub_tree_root + '\n')
                    fw.write(F'{PID}\n')
                    fw.write(F'EPOCH:{epoch}\n')
                    fw.write(F'NEWDEV:\n')
                    json.dump(status_dict[param_hash]['newest_dev'], fw, indent=4)
                    fw.write(F'\nBESTDEV:\n')
                    json.dump(status_dict[param_hash]['best_dev'], fw, indent=4)
                    fw.write('\n' + '=' * 70 + '\n')

        return
    else:
        for son in sons:
            recur(join(sub_tree_root, son), status_dict)


if __name__ == '__main__':

    root_path = join(path, root_folder)

    with codecs.open(json_file, 'w', 'utf8') as a:
        a.write('---Start---\n')

    # 维护一个全局的状态
    state = defaultdict(lambda: {'epoch': -1, 'newest_dev': [0.0, 0.0, 0.0], 'best_dev': [0.0, 0.0, 0.0]})

    while 1:
        recur(join(path, root_folder), state)
        sleep(1)



