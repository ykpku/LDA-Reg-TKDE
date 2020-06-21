#!/usr/bin/env Python
# coding=utf-8
import sys
import os
import numpy as np
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from params import MIMICP, EMBEDP
from utilities.csv_utility import CsvUtility


def reload_mimic_seq(train_percent=MIMICP.train_percent, valid=False, file_path=MIMICP.mimic_data_path, seq_num=MIMICP.seq_num):
    train_x = CsvUtility.read_sparse_array_from_csv(file_path, 'sparse_formal_train_x_seq.npz')
    train_x = train_x.reshape((train_x.shape[0], seq_num, -1))
    valid_x = CsvUtility.read_sparse_array_from_csv(file_path, 'sparse_formal_valid_x_seq.npz')
    valid_x = valid_x.reshape((valid_x.shape[0], seq_num, -1))
    test_x = CsvUtility.read_sparse_array_from_csv(file_path, 'sparse_formal_test_x_seq.npz')
    test_x = test_x.reshape((test_x.shape[0], seq_num, -1))
    train_y = CsvUtility.read_array_from_csv(file_path, 'formal_train_y_seq.csv')
    valid_y = CsvUtility.read_array_from_csv(file_path, 'formal_valid_y_seq.csv')
    test_y = CsvUtility.read_array_from_csv(file_path, 'formal_test_y_seq.csv')
    if valid:
        test_x = valid_x
        test_y = valid_y
    else:
        train_x = np.concatenate((train_x, valid_x), axis=0)
        train_y = np.concatenate((train_y, valid_y), axis=0)
    if train_percent < 0.8:
        new_training_size = int((train_x.shape[0] + test_x.shape[0]) * train_percent)
        train_x = train_x[:new_training_size]
        train_y = train_y[:new_training_size]
    return train_x, train_y, test_x, test_y

def get_some_instance(file_path=MIMICP.mimic_data_path, seq_num=MIMICP.seq_num, num=10):
    train_x = CsvUtility.read_sparse_array_from_csv(file_path, 'sparse_formal_train_x_seq.npz')
    train_x = train_x.reshape((train_x.shape[0], seq_num, -1))
    valid_x = CsvUtility.read_sparse_array_from_csv(file_path, 'sparse_formal_valid_x_seq.npz')
    valid_x = valid_x.reshape((valid_x.shape[0], seq_num, -1))
    test_x = CsvUtility.read_sparse_array_from_csv(file_path, 'sparse_formal_test_x_seq.npz')
    test_x = test_x.reshape((test_x.shape[0], seq_num, -1))
    train_y = CsvUtility.read_array_from_csv(file_path, 'formal_train_y_seq.csv')
    valid_y = CsvUtility.read_array_from_csv(file_path, 'formal_valid_y_seq.csv')
    test_y = CsvUtility.read_array_from_csv(file_path, 'formal_test_y_seq.csv')

    x_data = np.concatenate((train_x, valid_x, test_x), axis=0)
    y_data = np.concatenate((train_y, valid_y, test_y), axis=0)

    idx = np.random.permutation(x_data.shape[0])
    x_data = x_data[idx]
    y_data = y_data[idx]
    return x_data[:num], y_data[:num]

def reload_mimic_embedding(train_percent=MIMICP.train_percent, valid=False, file_path=MIMICP.mimic_data_path, seq_num=MIMICP.seq_num, embedding_type=EMBEDP.embedding_type, veclen=EMBEDP.veclen, window=EMBEDP.window):
    embedding_name = embedding_type + str(veclen) + '_window' + str(window)
    train_x = CsvUtility.read_array_from_csv(file_path, 'formal_train_x_seq_' + embedding_name + '.csv')
    train_x = train_x.reshape((train_x.shape[0], seq_num, -1))
    train_y = CsvUtility.read_array_from_csv(file_path, 'formal_train_y_seq_' + embedding_name + '.csv')
    valid_x = CsvUtility.read_array_from_csv(file_path, 'formal_valid_x_seq_' + embedding_name + '.csv')
    valid_x = valid_x.reshape((valid_x.shape[0], seq_num, -1))
    valid_y = CsvUtility.read_array_from_csv(file_path, 'formal_valid_y_seq_' + embedding_name + '.csv')
    test_x = CsvUtility.read_array_from_csv(file_path, 'formal_test_x_seq_' + embedding_name + '.csv')
    test_x = test_x.reshape((test_x.shape[0], seq_num, -1))
    test_y = CsvUtility.read_array_from_csv(file_path, 'formal_test_y_seq_' + embedding_name + '.csv')
    if valid:
        test_x = valid_x
        test_y = valid_y
    else:
        train_x = np.concatenate((train_x, valid_x), axis=0)
        train_y = np.concatenate((train_y, valid_y), axis=0)
    if train_percent < 0.8:
        new_training_size = int((train_x.shape[0] + test_x.shape[0]) * train_percent)
        train_x = train_x[:new_training_size]
        train_y = train_y[:new_training_size]
    return train_x, train_y, test_x, test_y
