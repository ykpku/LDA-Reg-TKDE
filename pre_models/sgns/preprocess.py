# -*- coding: utf-8 -*-

import os
import codecs
import pickle
import argparse
from gensim.corpora import Dictionary
import sys
from os import path
import numpy as np
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

# from utilities.csv_utility import CsvUtility
from lda import LdaTools
from params import LDAP

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/', help="data directory path")
    parser.add_argument('--vocab', type=str, default='./data/corpus.txt', help="corpus path for building vocab")
    parser.add_argument('--corpus', type=str, default='./data/corpus.txt', help="corpus path")
    parser.add_argument('--unk', type=str, default='<UNK>', help="UNK token")
    parser.add_argument('--window', type=int, default=5, help="window size")
    parser.add_argument('--max_vocab', type=int, default=20000, help="maximum number of vocab")
    return parser.parse_args()


class Preprocess(object):

    def __init__(self, window=5):
        self.window = window
        self.lda_tool = LdaTools(doc_path=LDAP.corpus_path, topic_num=LDAP.num_topic, PLSA=LDAP.plsa, corpus_file=LDAP.corpus_file, corpus_percent=LDAP.corpus_percent, output_path=LDAP.output_path, mimic_movie_wiki=LDAP.mimic0_movie1_wiki2)
        self.word_topic_dist = self.lda_tool.read_topic_distrib()
        if self.lda_tool.mimic_movie_wiki == 0:
            self.name = "MIMIC"
        elif self.lda_tool.mimic_movie_wiki == 1:
            self.name = "MovieReview"
        else:
            self.name = "Wiki"

    # def skipgram(self, sentence, i):
    #     iword = sentence[i]
    #     left = sentence[max(i - self.window, 0): i]
    #     right = sentence[i + 1: i + 1 + self.window]
    #     return iword, [self.unk for _ in range(self.window - len(left))] + left + right + [self.unk for _ in range(self.window - len(right))]

    def __get_cos_sim(self, x):
        x_stand = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        # x_stand = x
        mul = np.dot(x_stand, np.transpose(x_stand))
        mod = np.square(np.linalg.norm(x_stand, axis=1, keepdims=True))
        cos_sim = np.true_divide(mul, mod)
        return 0.5 + 0.5 * cos_sim


    def skipgram(self, i, n, sim):
        if n == 0:
            return None
        rank = np.argpartition(-sim[i, :], n)
        # print(i, set(rank[0:n+1]) - set([i]))
        if i in set(rank[0:n+1]):
            return i, set(rank[0:n+1]) - set([i])
        else:
            return i, set(rank[0:n])

    def build(self):
        print("building vocab...")

        filepath = self.lda_tool.doc_path + 'available_words_in_literature_' + self.name + '.csv'
        loaded_dct = Dictionary.load_from_text(filepath)
        print("---")
        self.idx2word = [ii[0] for ii in sorted(loaded_dct.token2id.items(), key=lambda d: d[1])]
        self.word2idx = loaded_dct.token2id
        # print(self.word2idx)
        # print(sorted(loaded_dct.token2id.items(), key=lambda d: d[1]))
        assert self.word2idx[self.idx2word[0]] == 0
        assert self.word2idx[self.idx2word[-1]] == len(self.idx2word)-1
        self.vocab = set(self.idx2word)
        self.wc = {}
        for w_id in range(len(self.idx2word)):
            self.wc[self.idx2word[w_id]] = loaded_dct.dfs[w_id]

        pickle.dump(self.wc, open(os.path.join(self.lda_tool.output_path, self.name + '_wc.dat'), 'wb'))
        pickle.dump(self.vocab, open(os.path.join(self.lda_tool.output_path, self.name + '_vocab.dat'), 'wb'))
        pickle.dump(self.idx2word, open(os.path.join(self.lda_tool.output_path, self.name + '_idx2word.dat'), 'wb'))
        pickle.dump(self.word2idx, open(os.path.join(self.lda_tool.output_path, self.name + '_word2idx.dat'), 'wb'))
        print("build done")


    def convert(self, filepath):
        print("converting corpus...")
        data = []
        cos_sim = self.__get_cos_sim(self.word_topic_dist)
        print(cos_sim[:5])
        for w_i in self.word2idx.values():
            iword, owords = self.skipgram(w_i, self.window, cos_sim)
            data.append((iword, list(owords)))
        print("")
        pickle.dump(data, open(os.path.join(self.lda_tool.output_path, self.name + '_train.dat'), 'wb'))
        print("conversion done")


if __name__ == '__main__':
    args = parse_args()
    preprocess = Preprocess(window=args.window)
    preprocess.build()
    preprocess.convert(args.corpus)

