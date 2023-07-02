import time
import torch
import numpy as np
from importlib import import_module
import argparse
import utils
import warnings
from plot_distri import plot_distribution
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser(description='Bruce-Bert-Text-Classsification')
parser.add_argument('--model', type=str, default='LSTM_GDAA', help = 'choose the model')
parser.add_argument('--transfer_method',type = str,default= "transfer_GDAA")
parser.add_argument('--num_class',type = int, default= 2)
args = parser.parse_args()

if __name__ == '__main__':
    torch.cuda.set_device(1)
    source_dataset = 'IMDB' #数据集地址
    target_dataset = 'SST2'

    model_name = args.model #上面几行的代码
    x = import_module('models.' + model_name) #在main中进行动态加载
    source_config = x.Config(dataset = source_dataset,seq_len =150 ,num_classes = args.num_class,transfer_to = target_dataset) # 调用数据集地址
    target_config = x.Config(dataset = target_dataset,seq_len = 50,num_classes = args.num_class,transfer_to = target_dataset)

    trans_method = import_module(args.transfer_method)
    #设随机种子

    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(4)
    torch.backends.cudnn.deterministic = True #保证每次运行结果一样

    start_time = time.time()
    print('加载数据集')
    train_data_src, dev_data_src, test_data_src,weight = utils.load_data(source_config) #返回数据集
    train_data_tar,dev_data_tar,test_data_tar,__ = utils.load_data(target_config)
    train_iter_src = utils.bulid_iterator(train_data_src, source_config) #将数据集变为可迭代的
    train_iter_tar = utils.bulid_iterator(train_data_tar, target_config)
    dev_iter_tar = utils.bulid_iterator(dev_data_tar, target_config)
    test_iter_tar = utils.bulid_iterator(test_data_tar,target_config)
    time_dif = utils.get_time_dif(start_time) #记录加载数据集的时间
    print("模型开始之前，准备数据时间：", time_dif)

    time_dif = utils.get_time_dif(start_time) #记录加载数据集的时间
    print("模型开始之前，准备数据时间：", time_dif)

    # 模型训练，评估与测试
    model = x.Model(weight,source_config).to(source_config.device) #动态加载的model 并将config对象送入模型中
    #model = torch.nn.DataParallel(model,device_ids=[3,2,1,0])
    trans_method.train(source_config, model, train_iter_src,
                        train_iter_tar,dev_iter_tar,test_iter_tar,dev_data_src,dev_data_tar)
