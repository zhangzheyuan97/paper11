import torch
import os
from utils import load_data
import numpy as np
import torch
import torch.nn as nn
import utils
import train
import time
import argparse
parser = argparse.ArgumentParser(description='Bruce-LSTM-Text-Classsification')
parser.add_argument('--model', type=str, default='LSTM')
args = parser.parse_args()
from importlib import import_module
import warnings
warnings.filterwarnings("ignore")
if __name__ == '__main__':
    dataset = 'IMDB' #数据集地址
    dataset_target = 'SST2'
    model_name = args.model #上面几行的代码
    x = import_module('models.' + model_name) #在main中进行动态加载
    config = x.Config(dataset = dataset,seq_len = 200,num_class = 2) #model.Bert 里面的调用 调用数据集地址
    #设随机种子
    config_target = x.Config(dataset = dataset_target,seq_len = 80,num_class = 2)
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True #保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data, dev_data, test_data,weight = utils.load_data(config) #返回数据集
    aa,bb,test_data_target,__ = utils.load_data(config_target)
    train_iter = utils.bulid_iterator(train_data, config) #将数据集变为可迭代的
    dev_iter = utils.bulid_iterator(dev_data, config)
    test_iter = utils.bulid_iterator(test_data, config)
    test_iter_tar = utils.bulid_iterator(test_data_target,config_target)
    time_dif = utils.get_time_dif(start_time) #记录加载数据集的时间
    print("模型开始之前，准备数据时间：", time_dif)

    # 模型训练，评估与测试
    model = x.Model(weight = weight,config = config).to(config.device) #动态加载的model 并将config对象送入模型中
    model = torch.nn.DataParallel(model)
    train.train(config, model, train_iter, dev_iter, test_iter,test_iter_tar)
    #train.test(config, model, test_iter)
