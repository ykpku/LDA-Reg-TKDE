import torch
import torch.utils.data as Data
from torch.autograd import Variable
import sys
from os import path
import numpy as np
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from utilities.metirc_utility import get_auc, get_auc_list, get_accuracy, get_precision_recall_f1

def test(x_test, y_test, net, nnp, out_name=""):
    test_dataset = Data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    test_loader = Data.DataLoader(dataset=test_dataset, batch_size=nnp.batchsize, shuffle=False, num_workers=2)
    res = []
    for data_iter in test_loader:
        tx, ty = data_iter
        if nnp.use_gpu:
            outputs = net(Variable(tx.cuda()).float())
            predicted = outputs.data.cpu()
        else:
            outputs = net(Variable(tx).float())
            predicted = outputs.data
        res.extend(list(predicted.numpy()))

    if nnp.num_classes == 1:
        auc_value = get_auc(y_test, res)
        auc_list = [auc_value]
    else:
        auc_list, _ = get_auc_list(y_test, res)

    accuracy_list, accuracy_mean = get_accuracy(y_test, res)
    precision_mean, recall_mean, f1_mean = get_precision_recall_f1(y_test, res)
    return auc_list, "%.6f" % accuracy_mean, "%.6f" % precision_mean, "%.6f" % recall_mean, "%.6f" % f1_mean