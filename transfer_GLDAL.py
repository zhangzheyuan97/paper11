#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
from sklearn import metrics
import time
from contrast_loss import SupConLoss
from plot_distri import plot_distribution
from cal_entropy import cal_entropy_weight
D_M = 0
D_C = 0
MU = 0
from floss import focal_loss
torch.cuda.set_device(1)
def pretrain(source_config, model, source_train_iter):
    print("start pretraining")
    pretrain_parameters = [
        {'params': model.module.atten.parameters(), "lr": source_config.pretrain_rate,"weight_decay":0.001},
        {'params': model.module.embedding.parameters(), 'lr': source_config.pretrain_rate, "weight_decay": 0.001},
        {'params': model.module.feature_extractor.parameters(), 'lr': source_config.pretrain_rate,
         "weight_decay": 0.001},
        {'params': model.module.linear.parameters(), 'lr': 0},
        {'params': model.module.dcis.parameters(), 'lr': 0},
        {'params': model.module.domain_classifier.parameters(), 'lr': 0}
    ]
    pretrain_optimizer = torch.optim.Adam(params=pretrain_parameters)
    criterion = SupConLoss(temperature=1)
    for epoch in range(10):
        model.train()
        totalloss = 0
        for i,src_all in enumerate(source_train_iter):
            (src_data, src_label) = src_all
            model.zero_grad()
            src_feature = model(src_data, None,transfer=False)
            loss = criterion(src_feature, src_label)
            totalloss = totalloss + loss.item()
            loss.backward()
            pretrain_optimizer.step()
        print("epoch:{} pretrain_loss:{}".format(epoch,totalloss))
def train(source_config, model, src_train_iter, tar_train_iter,tar_dev_iter, tar_test_iter,
          src_dev_dataset,tar_dev_dataset):
    """ source_config, target_config, model, src_train_iter,
                        tar_train_iter,tar_dev_iter,tar_test_iter
    模型训练方法
    :param config:
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    start_time = time.time()
    focal_criterion = focal_loss()
    #pretrain(source_config, model, source_train_iter= src_train_iter)
    optimizer_grouped_parameters = [
        {'params': model.atten.parameters(), "lr": source_config.learning_rate},
        {'params': model.embedding.parameters(), 'lr': source_config.learning_rate, "weight_decay": 0.00},
        {'params': model.feature_extractor.parameters(), 'lr': source_config.learning_rate,
         "weight_decay": 0.00},
        {'params': model.linear.parameters(), 'lr': source_config.learning_rate , "weight_decay": 0.0001},
        {'params': model.dcis.parameters(), 'lr': source_config.learning_rate,"weight_decay": 0.0001},
        {'params': model.domain_classifier.parameters(), 'lr': source_config.learning_rate,"weight_decay": 0.0001}
    ]

    optimizer = torch.optim.Adam(params= optimizer_grouped_parameters, lr=source_config.learning_rate)
    #启动 BatchNormalization 和 dropout
    global D_M,D_C,MU
    local_item = [0] * source_config.num_classes
    local_sigma = [1] * source_config.num_classes

    len_dataloader = len(src_train_iter)
    total_batch = 0 #记录进行多少batch
    dev_best_loss = float('inf') #记录校验集合最好的loss
    last_imporve = 0 #记录上次校验集loss下降的batch数 1000个batch后停止训练
    flag = False #记录是否很久没有效果提升，停止训练
    for epoch in range(source_config.num_epochs):
        model.train()
        print('Epoch [{}/{}'.format(epoch+1, source_config.num_epochs))

        if (D_M is 0) and (D_C is 0) and (MU is 0):
            MU = 0.5
        else:
            MU = 1- D_M / (D_M + D_C)
        d_m = 0
        d_c = 0
        d_c_list = [0] * source_config.num_classes
        print(1- MU," ",MU)
        print(local_sigma)
        for i, (src_all, tar_all) in enumerate(zip(src_train_iter,tar_train_iter)):
            (src_data,src_label) = src_all
            (tar_data, tar_label) = tar_all
            src_batch_size = src_label.size(0)
            tar_batch_size = tar_label.size(0)
            p = float(i + 1 + epoch * len_dataloader) / source_config.num_epochs/len_dataloader
            alpha = 2./(1. + np.exp(-10 * p)) - 1
            model.zero_grad()

            outputs = model(src_data,tar_data,alpha = alpha,transfer = True)
            s_output, t_output,s_domain_output,t_domain_output = outputs[0],outputs[1],outputs[2],outputs[3]
            s_local_out, t_local_out = outputs[4],outputs[5]
            #classifier loss
            class_loss = F.cross_entropy(s_output,src_label)

            #global_Domain_loss
            sdomain_entropy_weight = cal_entropy_weight(F.softmax(s_output,dim=1),source_config.num_classes)
            sdomain_label = torch.zeros(src_batch_size).long().to(source_config.device)
            err_s_domain = focal_criterion(s_domain_output,sdomain_label,sdomain_entropy_weight)
            tdomain_entropy_weight = cal_entropy_weight(F.softmax(t_output,dim= 1),source_config.num_classes)
            tdomain_label = torch.ones(tar_batch_size).long().to(source_config.device)
            err_t_domain = focal_criterion(t_domain_output,tdomain_label,tdomain_entropy_weight)

            #local_loss
            loss_s = 0.0
            loss_t = 0.0
            tmpd_c = 0.0


            for j in range(source_config.num_classes):
                loss_sj = focal_criterion(s_local_out[j],sdomain_label)
                loss_tj = focal_criterion(t_local_out[j],tdomain_label)
                loss_s += loss_sj * local_sigma[j]
                loss_t += loss_tj * local_sigma[j]
                d_c_list[j] += (2. * (0. - 2. * (loss_sj + loss_tj))).item()
                tmpd_c += 2. * (0. - 2. * (loss_sj + loss_tj))
            tmpd_c /= source_config.num_classes
            d_c = d_c + tmpd_c.item()

            global_loss = 0.1* (err_s_domain + err_t_domain)

            local_loss = 0.05 * (loss_s + loss_t)
            d_m = d_m + 2 * (0 - 2 * (err_s_domain + err_t_domain).item())
            join_loss = (1- MU) * global_loss + MU * local_loss
            loss = join_loss + class_loss
            loss.backward()
            optimizer.step()
            if total_batch % 50 == 0: #每多少次输出在训练集和校验集上的效果
                true = tar_label.data.cpu()
                predit = torch.max(t_output.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predit) #训练的准确度
                dev_acc, dev_loss = evaluate(source_config, model, tar_dev_iter) #在较验集上做测试
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), source_config.save_path) #保存模型
                    imporve = '*'
                    last_imporve = total_batch #last_improve以百为单位
                    #test(source_config, model, tar_test_iter)
                else:
                    imporve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.4}, Train Acc:{2:>6.2}, Val Loss:{3:>5.4}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, imporve))
            model.train()
            total_batch = total_batch + 1

            if total_batch - last_imporve > source_config.require_improvement:
                #在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break
        src_train_iter.index = 0
        tar_train_iter.index = 0
        D_M = np.copy(d_m).item()
        D_C = np.copy(d_c).item()
        for j in range(source_config.num_classes):
            local_sigma[j] = sum(d_c_list)/(source_config.num_classes * d_c_list[j])
        if flag:
            break
    test(source_config, model, tar_test_iter,src_dev_dataset,tar_dev_dataset)

def evaluate(config, model, dev_iter, test=False):
    """

    :param config:
    :param model:
    :param dev_iter:
    :return:
    """
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in dev_iter:
            outputs = model(texts,texts,transfer = True)[0]
            loss = F.cross_entropy(outputs, labels)
            loss_total = loss_total + loss
            labels = labels.data.cpu().numpy()
            predict = torch.max(outputs.data,1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)
    """
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(dev_iter), report, confusion
    """
    return acc, loss_total / len(dev_iter)


def test(config, model, test_iter,src_dev_dataset,tar_dev_dataset): #test中调用evaluate
    """
    模型测试
    :param config:
    :param model:
    :param test_iter:
    :return:
    """
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc,test_loss = evaluate(config, model, test_iter, test = True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    """
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    """
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：",time_dif)
    plot_distribution(model, src_dev_dataset, tar_dev_dataset)