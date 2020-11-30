import sys
from os import path
import os
import numpy as np
from sklearn import preprocessing
import pandas as pd
from sklearn.cluster import SpectralClustering
from collections import defaultdict
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from load_data.load_mimic_data import get_some_instance
from LDA_Reg import mlp_ldareg_gradient
from pre_models import lda
from utilities.csv_utility import CsvUtility
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

def look_neurons(path, file_name):
    neuron_grad = []
    with open(os.path.join(path, file_name), 'r')as f:
        for item in f.readlines():
            if item == "============================\n":
                pass
            else:
                line_sp = item.split("\t")
                # print(len(line_sp))
                neuron_grad.append(line_sp)
    return neuron_grad
# ng = look_neurons("E:\\garbage\\tkde_revision_temp_data\\interpretation\\", "test_neuron.csv")
# print(np.array(ng).shape)
# print(ng[:5])

def get_list_sort_index(data_list, pre_max_count):
    index_dict = {}
    for data_iter, data_item in enumerate(data_list):
        index_dict[data_iter] = data_item

    sort_items = sorted(index_dict.items(), key=lambda d: d[1], reverse=True)
    return [key for key, value in sort_items[:pre_max_count]], [value for key, value in sort_items[:pre_max_count]]

def get_neuron_label_count(train_y, neurons_gradient, max_neuron_num=3):
    neurons_label_dict = {}
    for patient_i, patient_y in enumerate(train_y):
        # print(np.sum(np.array(patient_y)), neurons_gradient[patient_i][patient_y == 1].shape, np.where(np.array(patient_y) == 1))
        labels_i = np.where(np.array(patient_y) == 1)[0]
        for ngs in neurons_gradient[patient_i][patient_y == 1]:
            n_ids, n_vals = get_list_sort_index(ngs, max_neuron_num)
            for n_id in n_ids:
                if n_id not in neurons_label_dict:
                    neurons_label_dict[n_id] = {}
                n_labels = neurons_label_dict[n_id]
                for label in labels_i:
                    n_labels[label] = n_labels.setdefault(label, 0) + 1
                neurons_label_dict[n_id] = n_labels
    return neurons_label_dict

def run_interpretation(base_path="/home1/yk/experiments_TKDE/major_revision/", model_file="#2020-11-21 03_33_04#_lstm_model_mimic_mlp_ldareg_1layer_0.8.pkl", num=10, neuron_num=3):
    train_x, train_y = get_some_instance(num=num)
    lda_tool = lda.LdaTools()
    net, sita, neuron_gradient = mlp_ldareg_gradient.train(train_x, train_y, lda_tool, path.join(base_path, model_file))
    neuron_gradient = np.abs(np.array(neuron_gradient))
    # print(neuron_gradient.shape)
    neuron_label_count = get_neuron_label_count(train_y, neuron_gradient, neuron_num)
    # print(neuron_label_count)
    print("patient", train_x.shape[0], "top neurons", neuron_num, "=======", len(neuron_label_count))
    return neuron_label_count

def test_patient_topneurons():
    need_results = []
    for p_i in range(50, 1000, 50):
        for n_i in range(2, 50, 2):
            if len(run_interpretation(num=p_i, neuron_num=n_i)) >= 128:
                print("need patient:", p_i)
                need_results.append(str(p_i) + "====" + str(n_i))
                break
    print(need_results)

def get_neron_labels(base_path, model_file, top_n_label, patient_num, neuron_num):
    neuron_label_count = run_interpretation(base_path=base_path, model_file=model_file, num=patient_num, neuron_num=neuron_num)
    neuron_labels = {}
    neuron_vals = {}
    for n_key, n_val in neuron_label_count.items():
        sorted_val = sorted(n_val.items(), key=lambda d: d[1], reverse=True)
        neuron_labels[n_key] = [k for k, v in sorted_val[:top_n_label]]
        neuron_vals[n_key] = [v for k, v in sorted_val[:top_n_label]]
        last_val = sorted_val[top_n_label-1][1]
        for f_k, f_v in sorted_val[top_n_label:]:
            if f_v == last_val:
                neuron_labels[n_key].append(f_k)
                neuron_vals[n_key].append(f_v)
            else:
                break
    return neuron_labels, neuron_vals

# #2020-11-21 03_33_04#_sita_mimic_mlp_ldareg_1layer_0.8.csv
def cluster_neurons(neuron_labels, base_path="/home1/yk/experiments_TKDE/major_revision/", sita_file="#2020-11-21 03_33_04#_sita_mimic_mlp_ldareg_1layer_0.8.csv", cluster_num=10):
    sita = CsvUtility.read_array_from_csv(base_path, sita_file)
    # print(sita[:3])
    # print(sita.shape)
    sc = SpectralClustering(cluster_num,  assign_labels='discretize', random_state=0)
    sc.fit(sita)
    # print(sc.labels_)
    label_cluster_matrix = np.zeros((80, cluster_num))
    for i, cluster in enumerate(sc.labels_):
        neuron_i_labels = neuron_labels[i]
        for nil in neuron_i_labels:
            label_cluster_matrix[nil][cluster] += 1
    return label_cluster_matrix, sc.labels_

def cal_class_entropy(label_cluster_matrix, neuron_num):
    # print("labels * clusters:")
    # print(label_cluster_matrix)
    # print(np.sum(label_cluster_matrix, axis=1, keepdims=True))
    pj_giv_i = label_cluster_matrix / (np.sum(label_cluster_matrix, axis=1, keepdims=True) + 1e-15)
    Ei = -1 * np.sum(pj_giv_i * np.log2(pj_giv_i + 1e-15), axis=1, keepdims=False)
    # print(pj_giv_i.shape, Ei.shape)
    E_sum = np.sum(Ei * np.sum(label_cluster_matrix, axis=1, keepdims=False)) / neuron_num
    print("class_entropy:", E_sum)
    return E_sum

def cal_F1(label_cluster_matrix, neuron_num, cluster_re):
    label_num = label_cluster_matrix.shape[0]
    cluster_num = label_cluster_matrix.shape[1]

    nj = np.zeros((1, cluster_num))
    for cri, c in enumerate(cluster_re):
        nj[0][c] += 1
    ni = np.sum(label_cluster_matrix, axis=1, keepdims=True)
    recall_matrix = label_cluster_matrix / nj
    precision_matrix = label_cluster_matrix / (ni + 1e-15)
    fij = 2 * precision_matrix * recall_matrix / (precision_matrix + recall_matrix + 1e-15)
    F = np.sum(ni * np.max(fij, axis=1, keepdims=True)) / neuron_num
    print("F1 :", F)
    return F

base_path = "/home1/yk/experiments_TKDE/major_revision/"
model_file = "#2020-11-21 03_33_04#_lstm_model_mimic_mlp_ldareg_1layer_0.8.pkl"
sita_file = "#2020-11-21 03_33_04#_sita_mimic_mlp_ldareg_1layer_0.8.csv"
r1, r2 = get_neron_labels(base_path=base_path, model_file=model_file, top_n_label=2, patient_num=100, neuron_num=10)
if len(r1) < 128:
    print("not cover all neurons !")
else:
    for tn in [2, 5, 10, 20, 30]:
        label_cluster_matrix, cluster_re = cluster_neurons(neuron_labels=r1, base_path=base_path, sita_file=sita_file, cluster_num=tn)
        cal_class_entropy(label_cluster_matrix=label_cluster_matrix, neuron_num=128)
        cal_F1(label_cluster_matrix=label_cluster_matrix, neuron_num=128, cluster_re=cluster_re)

