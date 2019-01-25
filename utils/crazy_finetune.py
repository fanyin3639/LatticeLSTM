# encoding: utf-8
"""
@author: wuwei 
@contact: wu.wei@pku.edu.cn

@version: 1.0
@license: Apache Licence
@file: crazy_finetune.py
@time: 19-1-2 下午9:50

写for循环疯狂调参
python main.py --highway  --nfeat 128 --use_wubi --gpu_id 3
"""
import logging
import os
from itertools import product

font_name = '/data/nfsdata/nlp/fonts/useful'

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('run.log')
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

# list里的第一个元素是默认设置
finetune_options = {
    'name': ['WeiboNER'],
    'mode': ['char'],
    'gaz_dropout': [0.5],
    'HP_lr': [0.01],
    'HP_dropout': [0.5],
    'HP_use_glyph': [True],
    'HP_glyph_ratio': [0.1, 0.001],
    'HP_font_channels': [2],
    'HP_glyph_highway': [False],
    'HP_glyph_embsize': [32],
    'HP_glyph_output_size': [32],
    'HP_glyph_dropout': [0.7],
    'HP_glyph_cnn_dropout': [0.5],
}


def construct_command(setting):
    command = 'python -m main'
    for feature, option in setting.items():
        if option is True:
            command += F' --{feature}'
        elif option is False:
            command += ''
        else:
            command += F' --{feature} {option}'
    return command


def traverse():
    """以默认配置为基准，每次只调一个参数，m个参数，每个参数n个选项，总共运行m*(n-1)次"""
    default_setting = {k: v[0] for k, v in finetune_options.items()}
    for feature in finetune_options:
        for i, option in enumerate(finetune_options[feature]):
            if i and default_setting[feature] != option:  # 默认设置
                setting = default_setting
                setting[feature] = option
                command = construct_command(setting)
                logger.info(command)
                try:
                    message = os.popen(command).read()
                except:
                    message = '进程启动失败!!'
                logger.info(message)


def grid_search():
    """以grid search的方式调参"""
    for vs in product(*finetune_options.values()):
        setting = {}
        for k, v in zip(finetune_options.keys(), vs):
            setting[k] = v
        command = construct_command(setting)
        logger.info(command)
        try:
            message = os.popen(command).read()
        except:
            message = '进程启动失败!!'
        logger.info(message)


if __name__ == '__main__':
    grid_search()
