# -*- coding:utf-8 _*-
""" 
@author:limuyu
@file: tiao.py 
@time: 2019/01/25
@contact: limuyu0110@pku.edu.cn

"""

from multiprocessing import Process, Manager, Lock
from time import sleep

from pynvml import *

from utils.crazy_finetune import *
options = {
    'name': ['CTB5POS', 'CTB6CWS', 'CTB6POS', 'CTB9POS', 'MSRANER', 'MSRCWS', 'NLPCCCWS', 'OntoNote4NER', 'PKUCWS',
             'ResumeNER', 'UD1POS', 'WeiboNER'],
    'mode': ['all'],
    'gaz_dropout': [0.5],
    'HP_lr': [0.01],
    'HP_dropout': [0.5, 0.3],
    'HP_use_glyph': [True],
    'HP_glyph_ratio': [0.1, 0.01, 0.001],
    'HP_font_channels': [1, 2, 8],
    'HP_glyph_highway': [False],
    'HP_glyph_embsize': [64],
    'HP_glyph_output_size': [64],
    'HP_glyph_dropout': [0.5, 0.3],
    'HP_glyph_cnn_dropout': [0.5, 0.3],
}


def judge_free_gpu(id):
    handle = nvmlDeviceGetHandleByIndex(id)
    info = nvmlDeviceGetMemoryInfo(handle)
    ps = nvmlDeviceGetComputeRunningProcesses(handle)
    if info.free / 1024 ** 2 > 3000 and len(ps) < 5:
        return True
    return False


def get_free_gpu_id_and_update(usage_list):
    useable = [10, 11, 12, 13, 14, 15]
    for i, a in enumerate(usage_list):
        if i in useable:
            if a < 3 and judge_free_gpu(i):
                usage_list[i] += 1
                return i
    return -1


def pao(gpu_id, command):
    command = command + F' --gpu_id {gpu_id}' + F' --command {command.replace(" " , "")}'
    print('=' * 70)
    print(command)
    print('=' * 70)
    try:
        message = os.popen(command).read()
        # print(command)
        # print(gpu_usage_list)
        # sleep(10 * random.random())
    except:
        message = '程启进败动了动失'
    logger.info(message)

    locktmp = Lock()
    locktmp.acquire()
    gpu_usage_list[gpu_id] -= 1
    locktmp.release()


if __name__ == '__main__':
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()
    gpu_usage_list = Manager().list([0 for i in range(deviceCount)])
    lock = Lock()
    params_list = list(product(*options.values()))
    while params_list:
        # sleep(2000)
        lock.acquire()
        gpu_id = get_free_gpu_id_and_update(gpu_usage_list)
        if gpu_id != -1:
            setting = {}
            param = params_list.pop()
            for k, v in zip(options.keys(), param):
                setting[k] = v
            command = construct_command(setting)
            logger.info(command)
            P = Process(target=pao, args=(gpu_id, command))
            P.start()
        lock.release()
        sleep(1)
