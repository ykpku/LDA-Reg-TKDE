import numpy as np
from os import path
import sys
import torch
import time
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from utilities.csv_utility import CsvUtility

def __split_metrics(result_epoch):
    metrics = []
    aucs_list = []
    for item in result_epoch:
        metrics.append([item[0], item[1][:4], item[2], item[3:]])
        # print(metrics)
        if len(item[1]) > 1:
            aucs_list.append(item[1][-1])
            # print(aucs_list[-1])
    return metrics, aucs_list



def save_results(net, sita, result_epoch, time_epochs, save_path, save_name):
    time_code = '#' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '#_'
    # save net
    torch.save(net, save_path + time_code + 'lstm_model_' + save_name + '.pkl')

    # save topic distribution
    CsvUtility.write_array2csv(sita, save_path, time_code + 'sita_' + save_name + '.csv')

    # save results
    metric_result, aucs = __split_metrics(result_epoch)
    CsvUtility.write_list2csv(metric_result, save_path, time_code + 'metrics_' + save_name + '.csv')
    if len(aucs) > 0:
        CsvUtility.write_array2csv(aucs, save_path, time_code + 'aucs_' + save_name + '.csv')

    # save time consuming
    CsvUtility.write_array2csv(time_epochs, save_path, time_code + 'time_epochs_' + save_name + '.csv')