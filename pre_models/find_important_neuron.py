import sys
from os import path
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd

sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from load_data.load_mimic_data import get_some_instance
from LDA_Reg import mlp_ldareg_gradient
from pre_models import lda
from utilities.csv_utility import CsvUtility
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

# have some bugs, correct in cluster_neuron_topics
def get_list_sort_index(data_list, pre_max_count):
    index_dict = {}
    # print 'original data length:', len(data_list)
    for data_iter, data_item in enumerate(data_list):
        index_dict[data_item] = data_iter
    # print 'dict data length:', len(index_dict)
    # print index_dict
    items = index_dict.items()
    items.sort(reverse=True)
    # print items
    return [key for key, value in items[:pre_max_count]], [value for key, value in items[:pre_max_count]]

def read_description():
    pd_diag = pd.read_csv('/home1/yk/experiments_TKDE/diagnoses_feature_seq.csv', sep='\t', index_col=None)
    pd_diag = pd_diag.set_index('feature_index')

    pd_re = pd.read_csv('/home1/yk/experiments_TKDE/diagnoses_target_seq.csv', sep='\t', index_col=None)
    pd_re = pd_re.set_index('pred_index')

    pd_lab = pd.read_csv('/home1/yk/experiments_TKDE/labtest_feature_seq.csv', sep='\t', index_col=None)
    pd_lab = pd_lab.set_index('feature_index')

    pd_pres = pd.read_csv('/home1/yk/experiments_TKDE/prescription_feature_seq.csv', sep='\t', index_col=None)
    pd_pres = pd_pres.set_index('feature_index')
    return pd_diag, pd_lab, pd_pres, pd_re

def read_topic(phi50, pd_diag, pd_lab, pd_pres, sita, chosed_neurons):
    neuron_td = sita[chosed_neurons]
    write_contend = []
    for l_iter, l_data in enumerate(neuron_td):
        keys, values = get_list_sort_index(l_data, 10)
        print '_______neuron ', chosed_neurons[l_iter], '_______'
        print 'keys:'
        print keys
        print 'values:'
        print values
        write_contend.append('_______neuron '+str(chosed_neurons[l_iter])+'_______')
        write_contend.append('keys:')
        write_contend.append(str(keys))
        write_contend.append('values:')
        write_contend.append(str(values))
        write_contend.extend(read_phi_feature(phi50, pd_diag, pd_lab, pd_pres, sp_topics=values))
    return write_contend

def read_phi_feature(phi50, pd_diag, pd_lab, pd_pres, sp_topics = [0], pre_word=20):
    write_contend = []
    for topic in sp_topics:
        keys, values = get_list_sort_index(phi50[topic], pre_word)
        print '--------------------------------------------------'
        print 'topic', topic, 'contend: '
        print keys
        print values
        write_contend.append('--------------------------------------------------')
        write_contend.append('topic'+str(topic)+'\t'+'contend: ')
        write_contend.append(str(keys))
        write_contend.append(str(values))
        for iter_w, word in enumerate(values):
            # print type(word)
            # print pd_diag.index
            if word in pd_diag.index:
                print 'Doiagnoses word:\t', keys[iter_w], np.array(pd_diag.ix[word])
                write_contend.append('Doiagnoses word:\t'+str(keys[iter_w])+'\t'+str(np.array(pd_diag.ix[word])))
            if word in pd_lab.index:
                print 'LabTest word:\t', keys[iter_w], np.array(pd_lab.ix[word])
                write_contend.append('LabTest word:\t'+str(keys[iter_w])+'\t'+str(np.array(pd_lab.ix[word])))
            if word in pd_pres.index:
                print 'Medication word:\t', keys[iter_w], np.array(pd_pres.ix[word])
                write_contend.append('Medication word:\t'+str(keys[iter_w])+'\t'+str(np.array(pd_pres.ix[word])))
        print '--------------------------------------------------'
        write_contend.append('--------------------------------------------------')
    return write_contend

def read_result(pd_re, train_y_line, i_line):
    write_contend = []
    re_list = []
    for p_i, i_re in enumerate(train_y_line):
        if i_re == 1:
            re_list.append(p_i)
    write_contend.append('---------------patient '+str(i_line)+' result---------------')
    write_contend.append(str(re_list))
    for item in re_list:
        write_contend.append('Result:\t'+str(np.array(pd_re.ix[item])))

    write_contend.append('--------------------------------------------------')
    return write_contend

def run():
    train_x, train_y = get_some_instance()
    lda_tool = lda.LdaTools()
    path = "/home1/yk/experiments_TKDE/#2020-01-07 03:21:12#_lstm_model_mimic_mlp_ldareg_1layer.pkl"
    net, sita, neuron_gradient = mlp_ldareg_gradient.train(train_x, train_y, lda_tool, path)
    neuron_gradient = np.array(neuron_gradient)
    neuron_gradient[neuron_gradient == float("inf")] = 0
    neuron_gradient[neuron_gradient == float("-inf")] = 0
    neuron_weight = np.sum(neuron_gradient, axis=2)
    min_max_scaler = preprocessing.MinMaxScaler()
    neuron_minmax = min_max_scaler.fit_transform(neuron_weight)

    # print(np.argmin(neuron_minmax, axis=0))
    # print(train_y)
    #
    # with open(os.path.join("/home1/yk/experiments_TKDE/", "test_neuron.csv"), 'w') as f:
    #     for item in neuron_gradient:
    #         f.write("============================\n")
    #         for line in item:
    #             for item in line:
    #
    #                 f.write(str(item) + '\t')
    #             f.write('\n')
    phi50 = CsvUtility.read_array_from_csv('/home1/yk/new_mimic_formal_data', 'MIMIC_phi_50.csv')
    pd_diag, pd_lab, pd_pres, pd_re = read_description()
    sita = CsvUtility.read_array_from_csv('/home1/yk/new_mimic_formal_data', 'sita_DK_medical_epoch600_pa10.0_pl1.0_lr0.001_wd0.0001_tp0.8_topics.csv')
    write_contend = []
    for patient_id, patient_neurons in enumerate(neuron_minmax):
        keys, values = get_list_sort_index(patient_neurons, 10)
        print '______patient ', patient_id, '__________'
        print 'keys:'
        print keys
        print 'values:'
        print values
        write_contend.append('______patient '+str(patient_id)+'__________')
        write_contend.append('keys:')
        write_contend.append(str(keys))
        write_contend.append('values:')
        write_contend.append(str(values))
        write_contend.extend(read_result(pd_re, train_y[patient_id], patient_id))
        write_contend.extend(read_topic(phi50, pd_diag, pd_lab, pd_pres, sita, values))

    CsvUtility.write_list2csv(write_contend, '/home1/yk/experiments_TKDE', 'interpresentation_1layer.csv')


run()
