import numpy as np
import sys
from os import path
import torch
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

def get_auc(Y, pred_res):
    fpr, tpr, thresholds = metrics.roc_curve(Y, pred_res)
    auc = metrics.auc(fpr, tpr)
    return "%.6f" % auc

def get_auc_list(y, pred_res):
    macro_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res), average='macro')
    micro_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res), average='micro')
    weight_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res), average='weighted')
    average_auc = metrics.roc_auc_score(np.array(y), np.array(pred_res))
    aucs = metrics.roc_auc_score(np.array(y), np.array(pred_res), average=None)
    return ["%.6f" % macro_auc, "%.6f" % micro_auc, "%.6f" % weight_auc, "%.6f" % average_auc, aucs], np.array(pred_res)

def get_accuracy(y, pred):
    acc_list = []
    y = np.array(y)
    pred = np.array(pred)
    pred_res = np.zeros(pred.shape)
    pred_res[pred >= 0.5] = 1.0
    pred_res[pred < 0.5] = 0.0
    if y.ndim == 1:
        acc_list.append(accuracy_score(y, pred_res))
    else:
        for col in range(y.shape[1]):
            acc_list.append(accuracy_score(y[:, col], pred_res[:, col]))
    return np.array(acc_list), np.mean(np.array(acc_list))

def get_accuracy_gpu(y, pred):
    pred_res = torch.zeros(pred.size()).cuda()
    pred_res[pred >= 0.5] = 1.0
    pred_res[pred < 0.5] = 0.0
    if y.dim() == 1:
        acc = 1.0 * (torch.sum(y == pred_res)) / y.size()[0]
        acc_list = torch.zeros((1)).cuda()
        acc_list[0] = acc
    else:
        acc_list = torch.zeros((y.size()[1])).cuda()
        for col in range(y.size()[1]):
            acc = 1.0 * (torch.sum(y[:, col] == pred_res[:, col])) / y.size()[0]
            acc_list[col] = acc
    return acc_list, torch.mean(acc_list)

def get_precision_recall_f1(y, pred):
    precision_list, recall_list, f1_list = [], [], []
    y = np.array(y)
    pred = np.array(pred)
    pred_res = np.zeros(pred.shape)
    pred_res[pred >= 0.5] = 1.0
    pred_res[pred < 0.5] = 0.0
    if y.ndim == 1:
        precision_list.append(precision_score(y, pred_res))
        recall_list.append(recall_score(y, pred_res))
        f1_list.append(f1_score(y, pred_res))
    else:
        for col in range(y.shape[1]):
            precision_list.append(precision_score(y[:, col], pred_res[:, col]))
            recall_list.append(recall_score(y[:, col], pred_res[:, col]))
            f1_list.append(f1_score(y[:, col], pred_res[:, col]))
    return np.mean(np.array(precision_list)), np.mean(np.array(recall_list)), np.mean(np.array(f1_list))


