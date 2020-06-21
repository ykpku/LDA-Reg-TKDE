import numpy as np
from os import path
import sys
import torch
import time
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from utilities.csv_utility import CsvUtility

from params import MIMICP, MOVIEP, EMBEDP, LSTMP, LDAP, ldaregP, MLPP

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



def save_results(net, sita, result_epoch, time_epochs, run_p):
    save_path = run_p.save_path
    save_name = run_p.save_name

    time_code = '#' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '#_'

    used_params = [run_p]
    if run_p.mimic0_movie1_wiki2 == 0:
        used_params.append(MIMICP)
    else:
        used_params.append(MOVIEP)
    if run_p.onehot0_embedding != 0:
        used_params.append(EMBEDP)
    if run_p.lm_lda_l2 == 0:
        used_params.append(LSTMP)
        used_params.append(LDAP)
        used_params.append(ldaregP)
    elif run_p.lm_lda_l2 == 1:
        used_params.append(LSTMP)
    elif run_p.lm_lda_l2 == 2:
        used_params.append(MLPP)
        used_params.append(LDAP)
        used_params.append(ldaregP)
    else:
        used_params.append(MLPP)
    for param_item in used_params:
        param_item.save_self(save_path, time_code + 'params_' + save_name + '.csv')

    # save net
    torch.save(net, save_path + time_code + 'lstm_model_' + save_name + '.pkl')

    # save topic distribution
    if sita.ndim > 1:
        CsvUtility.write_array2csv(sita, save_path, time_code + 'sita_' + save_name + '.csv')

    # save results
    metric_result, aucs = __split_metrics(result_epoch)
    CsvUtility.write_list2csv(metric_result, save_path, time_code + 'metrics_' + save_name + '.csv')
    if len(aucs) > 0:
        CsvUtility.write_array2csv(aucs, save_path, time_code + 'aucs_' + save_name + '.csv')

    # save time consuming
    CsvUtility.write_array2csv(time_epochs, save_path, time_code + 'time_epochs_' + save_name + '.csv')

