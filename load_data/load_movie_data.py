#!/usr/bin/env Python
# coding=utf-8
import sys
import os
from os import path
import numpy as np
from pprint import pprint
import argparse
from gensim.models import Word2Vec, FastText
sys.path.append(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0])

from params import MOVIEP, EMBEDP
from utilities.pickle_utility import read_pickle
from pre_models.Glove import glove2word2vec
from pre_models.generate_sgns_input import get_sgns_embedding


def reload_movie_seq(train_percent=MOVIEP.train_percent, valid=False, file_path=MOVIEP.movie_data_path, seq_num=MOVIEP.seq_num, veclen=MOVIEP.num_words):
    instance_data = read_pickle(file_path + 'movie_review_sequence_data.pkl', 'r')
    instance_result = read_pickle(file_path + 'movie_review_sequence_result.pkl', 'r')
    word2index = read_pickle(file_path + 'new_word2index.pkl', 'r')
    feature_tensor = np.zeros((len(instance_data), seq_num, veclen))
    for instance_iter, instance in enumerate(instance_data):
        start_index = seq_num - len(instance)
        for seq_iter, seq_data in enumerate(instance):
            code_index = word2index[seq_data]
            feature_tensor[instance_iter][seq_iter + start_index][code_index] += 1
    result_matrix = np.array(instance_result).reshape((len(instance_result), -1))
    train_size = int(feature_tensor.shape[0] * train_percent)
    train_x = feature_tensor[:train_size]
    train_y = result_matrix[:train_size]
    test_x = feature_tensor[train_size:]
    test_y = result_matrix[train_size:]
    if valid:
        new_train_size = int(train_size * train_percent)
        train_x = train_x[:new_train_size]
        train_y = train_y[:new_train_size]
        test_x = train_x[new_train_size:]
        test_y = train_y[new_train_size:]
    return train_x, train_y, test_x, test_y


def reload_movie_embedding(train_percent=MOVIEP.train_percent, valid=False, file_path=MOVIEP.movie_data_path, seq_num=MOVIEP.seq_num, embedding_type=EMBEDP.embedding_type, veclen=EMBEDP.veclen, window=EMBEDP.window):

    instance_data = read_pickle(file_path + 'movie_review_sequence_data.pkl', 'r')
    instance_result = read_pickle(file_path + 'movie_review_sequence_result.pkl', 'r')
    # word2index = read_pickle(file_path + 'new_word2index.pkl', 'r')
    feature_tensor = np.zeros((len(instance_data), seq_num, veclen))

    if embedding_type == 'embedding':
        model = Word2Vec.load(file_path + 'movie_review_word2vec_' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'embedding_skipgram':
        model = Word2Vec.load(file_path + 'movie_review_word2vec__skipgram' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'fasttext':
        model = FastText.load(file_path + 'movie_review_fasttext_' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'fasttext_skipgram':
        model = FastText.load(file_path + 'movie_review_fasttext__skipgram' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'glove':
        model = glove2word2vec(file_path + 'movie_vectors_w'+str(window)+'_l'+str(veclen)+'.txt', file_path + 'glove' + str(veclen)+'_window'+str(window)+'.model')
    elif embedding_type == 'lda_sgns' or embedding_type == 'sg_add_sgns' or embedding_type == 'sg_cancat_sgns':
        model = get_sgns_embedding('MovieReview')

    for instance_iter, instance in enumerate(instance_data):
        start_index = seq_num - len(instance)
        for seq_iter, seq_data in enumerate(instance):
            word_vec = model[seq_data]
            feature_tensor[instance_iter][seq_iter + start_index] += word_vec
    result_matrix = np.array(instance_result).reshape((len(instance_result), -1))
    train_size = int(feature_tensor.shape[0] * train_percent)
    train_x = feature_tensor[:train_size]
    train_y = result_matrix[:train_size]
    test_x = feature_tensor[train_size:]
    test_y = result_matrix[train_size:]
    if valid:
        new_train_size = int(train_size * train_percent)
        train_x = train_x[:new_train_size]
        train_y = train_y[:new_train_size]
        test_x = train_x[new_train_size:]
        test_y = train_y[new_train_size:]

    if embedding_type == 'sg_add_sgns' or embedding_type == 'sg_cancat_sgns':
        train_x_sg, train_y_sg, test_x_sg, test_y_sg = reload_movie_embedding(train_percent=train_percent, valid=valid, file_path=file_path, seq_num=seq_num, embedding_type="embedding_skipgram", veclen=veclen, window=window)
        if embedding_type == 'sg_add_sgns':
            train_x = train_x + train_x_sg
            test_x = test_x + test_x_sg
        if embedding_type == 'sg_cancat_sgns':
            train_x = np.concatenate((train_x, train_x_sg), axis=2)
            test_x = np.concatenate((test_x, test_x_sg), axis=2)

    return train_x, train_y, test_x, test_y


def reload_wiki_embedding(train_percent=MOVIEP.train_percent, valid=False, file_path=MOVIEP.movie_data_path, seq_num=MOVIEP.seq_num, embedding_type=EMBEDP.embedding_type, veclen=EMBEDP.veclen, window=EMBEDP.window):
    instance_data = read_pickle(file_path + 'movie_review_sequence_data.pkl', 'r')
    instance_result = read_pickle(file_path + 'movie_review_sequence_result.pkl', 'r')
    # word2index = read_pickle(file_path + 'new_word2index.pkl', 'r')
    feature_tensor = np.zeros((len(instance_data), seq_num, veclen))

    if embedding_type == 'embedding':
        model = Word2Vec.load(file_path + 'wiki_word2vec_' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'embedding_skipgram':
        model = Word2Vec.load(file_path + 'wiki_word2vec_skipgram' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'fasttext':
        model = FastText.load(file_path + 'wiki_fasttext_' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'fasttext_skipgram':
        model = FastText.load(file_path + 'wiki_fasttext_skipgram' + str(veclen) + '_window' + str(window) + '.model')
    elif embedding_type == 'glove':
        model = glove2word2vec(file_path + 'wiki_vectors_w'+str(window)+'_l'+str(veclen)+'.txt', file_path + 'glove' + str(veclen)+'_window'+str(window)+'.model')
    elif embedding_type == 'lda_sgns' or embedding_type == 'sg_add_sgns' or embedding_type == 'sg_cancat_sgns':
        model = get_sgns_embedding('Wiki')
    miss_count = 0
    for instance_iter, instance in enumerate(instance_data):
        start_index = seq_num - len(instance)
        for seq_iter, seq_data in enumerate(instance):
            if seq_data in model:
                word_vec = model[seq_data]
                feature_tensor[instance_iter][seq_iter + start_index] += word_vec
            else:
                miss_count += 1
    # print "----", miss_count, "----"
    result_matrix = np.array(instance_result).reshape((len(instance_result), -1))
    train_size = int(feature_tensor.shape[0] * train_percent)
    train_x = feature_tensor[:train_size]
    train_y = result_matrix[:train_size]
    test_x = feature_tensor[train_size:]
    test_y = result_matrix[train_size:]
    if valid:
        new_train_size = int(train_size * train_percent)
        train_x = train_x[:new_train_size]
        train_y = train_y[:new_train_size]
        test_x = train_x[new_train_size:]
        test_y = train_y[new_train_size:]

    if embedding_type == 'sg_add_sgns' or embedding_type == 'sg_cancat_sgns':
        train_x_sg, train_y_sg, test_x_sg, test_y_sg = reload_wiki_embedding(train_percent=train_percent, valid=valid, file_path=file_path, seq_num=seq_num, embedding_type="embedding_skipgram", veclen=veclen, window=window)
        if embedding_type == 'sg_add_sgns':
            train_x = train_x + train_x_sg
            test_x = test_x + test_x_sg
        if embedding_type == 'sg_cancat_sgns':
            train_x = np.concatenate((train_x, train_x_sg), axis=2)
            test_x = np.concatenate((test_x, test_x_sg), axis=2)
    return train_x, train_y, test_x, test_y

