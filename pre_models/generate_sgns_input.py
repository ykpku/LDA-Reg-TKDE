#!/usr/bin/env Python
# coding=utf-8
import sys
import os
from os import path
import numpy as np
from pprint import pprint
import argparse
import pickle

sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])
from lda import LdaTools
from utilities.csv_utility import CsvUtility
from com.word_id_dict import WordIndexMap
from params import LDAP, MIMICP, MOVIEP, EMBEDP


def __filter_events(all_events):
    # 90 days into 9 windows, each window has 10 days.
    # how many empty sequence and delete sequence which has no event
    seq_sum = 0
    filter_event = []
    count_zero = 0
    count_one = 0
    for event_pred in all_events:
        if len(event_pred) != 0:
            event = event_pred[0]
            ec = 0
            for ev_item in event:
                if len(ev_item) != 0:
                    ec = ec + 1
            if ec == 0:
                count_zero = count_zero + 1
                continue
            if ec == 1:
                count_one = count_one + 1
                # continue
            pred = event_pred[1]
            seq_sum = seq_sum + ec
            filter_event.append(event_pred)
        else:
            count_zero = count_zero + 1
    print 'seq mean : ', seq_sum / len(filter_event)
    print 'count zero :', count_zero
    print 'count one :', count_one
    print 'filter events :', len(filter_event)
    '''
    # 90 time window split into 9 sequence, each sequence has 10 days
    seq mean :  3
    count zero : 635
    count one : 9473
    '''
    return filter_event


def get_sgns_embedding(name, sgns_idx2vec_path=LDAP.output_path):
    # if LDAP.mimic0_movie1_wiki2 == 0:
    #     name = "MIMIC"
    # elif LDAP.mimic0_movie1_wiki2 == 1:
    #     name = "MovieReview"
    # else:
    #     name = "Wiki"
    idx2word = pickle.load(open(os.path.join(sgns_idx2vec_path, name + '_idx2word.dat'), 'rb'))
    idx2vec = pickle.load(open(os.path.join(sgns_idx2vec_path, name + '_idx2vec.dat'), 'rb'))
    assert len(idx2vec) == len(idx2word)
    # print(idx2word)
    # print("----")
    # print(idx2vec)
    word2vec = {}
    for idx in range(len(idx2word)):
        word2vec[idx2word[idx]] = idx2vec[idx]
    return word2vec


def get_mimic_sequence_data(data_pickle_path, word_dict_path, predict_dict_path, seq_max, vec_len, sgns_path, save_path, save=False):
    all_events = CsvUtility.read_pickle(data_pickle_path, 'r')
    word_dict = CsvUtility.read_pickle(word_dict_path, 'r')
    predict_dict = CsvUtility.read_pickle(predict_dict_path, 'r')
    # pprint(all_events[0])
    print "word_dict:", len(word_dict), "predict_dict:", len(predict_dict), "all_events:", len(all_events)
    feature_dict = WordIndexMap(list(word_dict))
    pred_dict = WordIndexMap(list(predict_dict))

    filter_event = __filter_events(all_events=all_events)
    sgns_model = get_sgns_embedding('MIMIC', sgns_path)

    feature_tensor = np.zeros((len(filter_event), seq_max, vec_len))
    feature_count_tensor = np.zeros((len(filter_event), seq_max))
    result_tensor = np.zeros((len(filter_event), len(predict_dict)))

    find_nan = {}
    for i_iter, event_line in enumerate(filter_event):
        for seq_iter, sequence_item in enumerate(event_line[0]):
            for event_code in sequence_item:
                if event_code in sgns_model:

                    feature_tensor[i_iter][seq_iter] += sgns_model[event_code]
                    feature_count_tensor[i_iter][seq_iter] += 1
                else:
                    if event_code in find_nan:
                        find_nan[event_code] += 1
                    else:
                        find_nan[event_code] = 1
        for pred_item in event_line[1]:
            result_tensor[i_iter][pred_dict.get_index_by_word(pred_item)] = 1

        if i_iter % 1000 == 0:
            print 'complete {0} of {1}'.format(i_iter, len(filter_event))
    print 'words not in docs:', len(find_nan)
    if save:
        CsvUtility.write_dict2csv(feature_dict.get_word2index(), save_path, 'feature2index_seq_embedding'+str(vec_len)+'.csv')
        CsvUtility.write_dict2csv(pred_dict.get_word2index(), save_path, 'predict2index_seq_embedding'+str(vec_len)+'.csv')
        CsvUtility.write_array2csv(feature_tensor.reshape((feature_tensor.shape[0], -1)), save_path, 'feature_matrix_seq_embedding'+str(vec_len)+'.csv')
        CsvUtility.write_array2csv(result_tensor.reshape((result_tensor.shape[0], -1)), save_path, 'result_matrix_seq_embedding'+str(vec_len)+'.csv')

    return feature_tensor, feature_count_tensor, result_tensor

def get_movie_sequence_data(data_pickle_path, word_dict_path, predict_dict_path, seq_max, vec_len, sgns_path, save_path, save=False):
    pass

def __get_aggregate_seq(feature_tensor, feature_count_tensor, seq_not):
    if seq_not:
        feature_count_tensor[feature_count_tensor == 0] = 1.0
        feature_count_tensor = feature_count_tensor.reshape((feature_count_tensor.shape[0], feature_count_tensor.shape[1], 1))
        feature_tensor = feature_tensor * 1.0 / feature_count_tensor
    else:
        feature_tensor = np.sum(feature_tensor, axis=1) / np.sum(feature_count_tensor, axis=1, keepdims=True)
    return feature_tensor

def generate_train_test(base_path, seq_max, vec_len, sgns_path, save_path, seq_not, train_valid_perc=0.8, shuffle=False, save=False):

    feature_tensor, feature_count_tensor, result_tensor = get_mimic_sequence_data(data_pickle_path=base_path+'after_instance_seq.pkl',
                                                            word_dict_path=base_path+'event_instance_dict_seq.pkl',
                                                            predict_dict_path=base_path+'predict_diags_dict_seq.pkl',
                                                            seq_max=seq_max,
                                                            vec_len=vec_len,
                                                            sgns_path=sgns_path,
                                                            save_path=save_path,
                                                            save=False)
    feature_tensor = __get_aggregate_seq(feature_tensor, feature_count_tensor, seq_not)
    x = feature_tensor.reshape((feature_tensor.shape[0], -1))
    y = result_tensor.reshape((result_tensor.shape[0], -1))
    train_size = int(x.shape[0] * train_valid_perc)
    # for further extention
    name_append = 'SGNS'
    # shuffle the train set
    if shuffle:
        idx = np.random.permutation(x.shape[0])
        CsvUtility.write_array2csv(idx, base_path, 'random_idx_seq_' + name_append + '.csv')
    else:
        idx = CsvUtility.read_array_from_csv(base_path, 'random_idx_seq_' + name_append + '.csv')
    x_train = x[idx]
    y_train = y[idx]

    training_x = x_train[:train_size]
    training_y = y_train[:train_size]
    testing_x = x_train[train_size:]
    testing_y = y_train[train_size:]
    # print training_x.shape
    # print training_y.shape
    # print testing_x.shape
    # print testing_y.shape
    # print len(idx)
    if save:
        CsvUtility.write_array2csv(training_x, save_path, 'formal_train_valid_x_seq_'+name_append+'.csv')
        CsvUtility.write_array2csv(training_y, save_path, 'formal_train_valid_y_seq_'+name_append+'.csv')
        CsvUtility.write_array2csv(testing_x, save_path, 'formal_test_x_seq_'+name_append+'.csv')
        CsvUtility.write_array2csv(testing_y, save_path, 'formal_test_y_seq_'+name_append+'.csv')
    return training_x, training_y, testing_x, testing_y


def get_train_validation_test_seq(base_path, seq_max, vec_len, sgns_path, save_path, seq_not, train_perc=0.8, shuffle=False, save=False):
    training_x, training_y, testing_x, testing_y = generate_train_test(base_path, seq_max, vec_len, sgns_path, save_path, seq_not, train_perc, shuffle, save)
    training_size = int(training_x.shape[0] * 0.8)
    formal_training_x = training_x[:training_size]
    formal_training_y = training_y[:training_size]
    validation_x = training_x[training_size:]
    validation_y = training_y[training_size:]
    print formal_training_x.shape
    print formal_training_y.shape
    print validation_x.shape
    print validation_y.shape
    print testing_x.shape
    print testing_y.shape
    # for further extention
    embedding_append = 'lda_sgns500_window50'

    CsvUtility.write_array2csv(formal_training_x, save_path, 'formal_train_x_seq_' + embedding_append + '.csv')
    CsvUtility.write_array2csv(formal_training_y, save_path, 'formal_train_y_seq_' + embedding_append + '.csv')
    CsvUtility.write_array2csv(validation_x, save_path, 'formal_valid_x_seq_' + embedding_append + '.csv')
    CsvUtility.write_array2csv(validation_y, save_path, 'formal_valid_y_seq_' + embedding_append + '.csv')
    CsvUtility.write_array2csv(testing_x, save_path, 'formal_test_x_seq_' + embedding_append + '.csv')
    CsvUtility.write_array2csv(testing_y, save_path, 'formal_test_y_seq_' + embedding_append + '.csv')
    return training_x, training_y, validation_x, validation_y, testing_x, testing_y


if __name__ == '__main__':

    if LDAP.mimic0_movie1_wiki2 == 0:
        base_path = MIMICP.mimic_data_path
        seq_max = MIMICP.seq_num
        train_perc = MIMICP.train_percent
    else:
        base_path = MOVIEP.movie_data_path
        seq_max = MOVIEP.seq_num
        train_perc = MOVIEP.train_percent
    vec_len = EMBEDP.veclen
    sgns_path = LDAP.output_path
    save_path = LDAP.output_path
    seq_not = True
    shuffle = True
    save = False
    x_train, y_train, x_valid, y_valid, x_test, y_test = get_train_validation_test_seq(
        base_path, seq_max, vec_len, sgns_path, save_path, seq_not, train_perc, shuffle, save
    )
