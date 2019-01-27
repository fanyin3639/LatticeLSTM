# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: test_all.py
@time: 19-1-27 上午11:33


"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
args = parser.parse_args()


def test_all(path):
    for p in os.listdir(path):
        q = os.path.join(path, p)
        command = F'python main.py --status test --loadmodel {q}'
        try:
            os.system(command)
        except:
            print(command + 'failed')


def recur_test(sub_tree_root, status_dict=None):
    if os.path.isdir(sub_tree_root):
        sons = os.listdir(sub_tree_root)
        if 'data.set' in sons:
            command = F'python ../main.py --status test --loadmodel {sub_tree_root}'
            try:
                os.system(command)
            except:
                print(command + 'failed')
        else:
            for son in sons:
                recur_test(os.path.join(sub_tree_root, son), status_dict)


if __name__ == '__main__':
    for sub in os.listdir(args.path):
        print('*'*20)
        print(F'start decoding {sub}')
        print('*'*20)
        recur_test(os.path.join(args.path, sub))
