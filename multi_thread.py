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
import re

# src_folder会在/data/nfsdata/nlp/projects底下创建树根
src_folder = 'MUYUWEIBO'

# 选择GPU的策略，大于下述数值才会选择 (MB)
min_gpu_free_mem = 3500

# 选择GPU的策略，GPU上面的进程数目小于下述数值才会选择
max_gpu_process = 5

# 在每一块GPU上最多并发跑这个程序的数量
max_mine_process = 1

# 使用的GPU编号
gpus = [2]

options = {
    'name': ['UD1POS'],
    'mode': ['char'],
    'gaz_dropout': [0.5],
    'HP_lr': [0.01],
    'HP_dropout': [0.5],
    'HP_use_glyph': [True],
    'HP_glyph_ratio': [0.1, 0.01, 0.001],
    'HP_font_channels': [1, 2, 4, 8],
    'HP_glyph_highway': [False],
    'HP_glyph_embsize': [64],
    'HP_glyph_output_size': [64],
    'HP_glyph_cnn_dropout': [0.7, 0.5, 0.3],
}


def judge_free_gpu(id):
    handle = nvmlDeviceGetHandleByIndex(id)
    info = nvmlDeviceGetMemoryInfo(handle)
    ps = nvmlDeviceGetComputeRunningProcesses(handle)
    if info.free / 1024 ** 2 > min_gpu_free_mem and len(ps) < max_gpu_process:
        return True
    return False


def get_free_gpu_id_and_update(usage_list):
    for i, a in enumerate(usage_list):
        if i in gpus:
            if a < max_mine_process and judge_free_gpu(i):
                usage_list[i] += 1
                return i
    return -1


def pao(gpu_id, command_in, setting_str):
    commandline = command + F' --gpu_id {gpu_id}' + F' --setting_str {setting_str}+PID.{os.getpid()}+PPID.{os.getppid()} --src_folder {src_folder}'

    print(commandline)
    print('=' * 70)
    try:
        message = os.popen(commandline).read()
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
            setting_str = re.sub(r"[\'\{\}\s]", '', str(setting))
            setting_str = setting_str.replace(',', '/').replace(':', '.')
            P = Process(target=pao, args=(gpu_id, command, setting_str))
            P.start()
        lock.release()
        sleep(1)
