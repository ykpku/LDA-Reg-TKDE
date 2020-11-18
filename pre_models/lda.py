from os import path
import pandas as pd
import numpy as np
import argparse
from gensim import corpora, models
import sys
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from utilities.csv_utility import CsvUtility
from params import LDAP, MOVIEP, MIMICP
from pre_models.build_wikipedia_data import get_filter_data


class LdaTools(object):
    def __init__(self, doc_path, corpus_file, topic_num, PLSA, corpus_percent, output_path, mimic_movie_wiki):
        self.doc_path = doc_path
        self.topic_num = topic_num
        self.dictionary = []
        self.lda_model = None
        self.PLSA = PLSA
        self.corpus_file = corpus_file
        self.corpus_percent = corpus_percent
        self.output_path = output_path
        self.mimic_movie_wiki = mimic_movie_wiki

    def train_lda(self, passes=5):
        if self.mimic_movie_wiki == 0:
            name = "_MIMIC"
        elif self.mimic_movie_wiki == 1:
            name = "_MovieReview"
        else:
            name = "_Wiki"
        if self.mimic_movie_wiki == 0 or self.mimic_movie_wiki == 1:
            selected_docs = pd.read_csv(self.doc_path+self.corpus_file, header=None, index_col=[0]).values
            size = int(len(selected_docs) * self.corpus_percent)
            selected_docs = selected_docs[:size]
            texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
        else:
            texts = get_filter_data(self.doc_path)
            size = int(len(texts) * self.corpus_percent)
            texts = texts[:size]
        self.dictionary = corpora.Dictionary(texts)
        self.dictionary.save_as_text(self.doc_path+'available_words_in_literature'+name+'.csv')
        # print self.dictionary
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]
        print 'number of docs:', len(self.corpus)
        if self.PLSA:
            alpha_str = 'asymmetric'
        else:
            alpha_str = 'auto'
        self.lda_model = models.LdaModel(self.corpus, alpha=alpha_str, id2word=self.dictionary, num_topics=self.topic_num, passes=passes)
        print 'lda training end.'

    def get_alpha(self):
        return self.lda_model.alpha

    def get_theta(self):
        theta_re = np.zeros([len(self.corpus), self.topic_num])

        for text_i, text in enumerate(self.corpus):
            for t_i in self.lda_model[text]:
                theta_re[text_i][t_i[0]] += t_i[1]
        return theta_re

    # shape of (topics, words)
    def get_mimic_phi(self, feature2index_file):
        lda_phi = self.lda_model.get_topics()
        return self.__change_word_index__(lda_phi, 1, feature2index_file)

    def get_movie_phi(self, word2index_file):
        lda_phi = self.lda_model.get_topics()
        return self.__change_Movie_word_index__(lda_phi, word2index_file)

    def show_topics(self, topic_n, word_n):
        print(self.lda_model.show_topics(topic_n, word_n))

    def __change_Movie_word_index__(self, gamma, word2index_pickle):
        feature_word2id = CsvUtility.read_pickle(word2index_pickle, 'r')
        print 'feature size: ', len(feature_word2id)
        change_index_result = np.zeros((gamma.shape[0], len(feature_word2id)))
        for j in range(gamma.shape[1]):
            new_index = feature_word2id[self.dictionary.__getitem__(j)]
            for i in range(gamma.shape[0]):
                change_index_result[i][new_index] += gamma[i][j]
            if j % 1000 == 0:
                print j, 'line'
        print 'after changing the size of result: ', change_index_result.shape
        return change_index_result

    def __change_word_index__(self, gamma, idx, feature2index_path):
        print 'dictionary size: ', self.dictionary.__len__()
        feature_word2id = {}
        feature_index = pd.read_csv(feature2index_path,header=None, index_col=None)
        f_i = np.array(feature_index)
        for i in range(f_i.shape[0]):
            feature_word2id[f_i[i][0]] = int(f_i[i][1])
        print 'feature size: ', len(feature_word2id)

        if idx == 0:
            change_index_result = np.zeros((feature_index.shape[0], gamma.shape[1]))
            for i in range(gamma.shape[0]):
                new_index = feature_word2id[self.dictionary.__getitem__(i)]
                for j in range(gamma.shape[1]):
                    change_index_result[new_index][j] += gamma[i][j]
                if i % 1000 == 0:
                    print i, 'line'
            print 'after changing the size of result: ', change_index_result.shape
        else:
            change_index_result = np.zeros((gamma.shape[0], feature_index.shape[0]))
            for j in range(gamma.shape[1]):
                new_index = feature_word2id[self.dictionary.__getitem__(j)]
                for i in range(gamma.shape[0]):
                    change_index_result[i][new_index] += gamma[i][j]
                if j % 1000 == 0:
                    print j, 'line'
            print 'after changing the size of result: ', change_index_result.shape
        return change_index_result

    def save_phi_alpha_theta_topicdistrib(self):
        plsa=""
        percent=""
        if self.PLSA:
            plsa = "PLSA"
        if self.corpus_percent != 1.0:
            percent = "_" + str(self.corpus_percent)+"percent"
        if self.mimic_movie_wiki == 0:
            CsvUtility.write_array2csv(self.get_alpha(), self.output_path, 'MIMIC_alpha_'+str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_mimic_phi(MIMICP.feature_index_file), self.output_path, 'MIMIC_phi_'+str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_theta(), self.output_path, 'MIMIC_theta_'+str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_topic_distrib_of_word(), self.output_path, 'MIMIC_topic_distrib_'+str(self.topic_num) + plsa + percent + '.csv')
        elif self.mimic_movie_wiki == 1:
            CsvUtility.write_array2csv(self.get_alpha(), self.output_path, 'MovieReview_alpha_' + str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_movie_phi(MOVIEP.feature_index_file), self.output_path, 'MovieReview_phi_' + str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_theta(), self.output_path, 'MovieReview_theta_' + str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_topic_distrib_of_word(), self.output_path, 'MovieReview_topic_distrib_' + str(self.topic_num) + plsa + percent + '.csv')
        else:
            CsvUtility.write_array2csv(self.get_alpha(), self.output_path, 'Wiki_alpha_' + str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_movie_phi(MOVIEP.feature_index_file), self.output_path, 'Wiki_phi_' + str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_theta(), self.output_path, 'Wiki_theta_' + str(self.topic_num) + plsa + percent + '.csv')
            CsvUtility.write_array2csv(self.get_topic_distrib_of_word(), self.output_path, 'Wiki_topic_distrib_' + str(self.topic_num) + plsa + percent + '.csv')

    def __read_phi_alpha_theta_byname(self, name):
        plsa = ""
        percent = ""
        if self.PLSA:
            plsa = "_PLSA"
        if self.corpus_percent != 1.0:
            percent = "_" + str(self.corpus_percent)+"percent"
        alpha = CsvUtility.read_array_from_csv(self.output_path, name+'_alpha_' + str(self.topic_num) + plsa + percent + '.csv')
        phi = CsvUtility.read_array_from_csv(self.output_path, name+'_phi_' + str(self.topic_num) + plsa + percent + '.csv')
        theta = CsvUtility.read_array_from_csv(self.output_path, name+'_theta_' + str(self.topic_num) + plsa + percent + '.csv')
        return alpha, phi, theta

    def read_phi_alpha_theta(self):
        if self.mimic_movie_wiki == 0:
            return self.__read_phi_alpha_theta_byname("MIMIC")
        elif self.mimic_movie_wiki == 1:
            return self.__read_phi_alpha_theta_byname("MovieReview")
        else:
            return self.__read_phi_alpha_theta_byname("Wiki")

    def read_topic_distrib(self):
        if self.mimic_movie_wiki == 0:
            name = "MIMIC"
        elif self.mimic_movie_wiki == 1:
            name = "MovieReview"
        else:
            name = "Wiki"
        plsa = ""

        if self.PLSA:
            plsa = "_PLSA"
        percent = ""
        if self.corpus_percent != 1.0:
            percent = "_" + str(self.corpus_percent) + "percent"
        topic_distrib = CsvUtility.read_array_from_csv(self.output_path, name + '_topic_distrib_' + str(self.topic_num) + plsa + percent + '.csv')
        return topic_distrib

    def __get_word_count(self):
        wc = np.zeros([len(self.corpus), len(self.dictionary)])
        for doc_i, doc in enumerate(self.corpus):
            for item in doc:
                wc[doc_i][item[0]] += item[1]
        return wc

    def get_topic_distrib_of_word(self):
        lda_phi = self.lda_model.get_topics().transpose() # K * V-- V * K
        pzi = np.array(self.get_theta()).mean(axis=0, keepdims=True) # D * K
        pwj_tem = self.__get_word_count().sum(axis=0, keepdims=True).transpose() # D * V -- V * D
        pwj = pwj_tem / pwj_tem.sum()
        re = lda_phi * pzi / pwj
        return re


# if __name__ == '__main__':
#     lt = LdaTools(doc_path=LDAP.corpus_path, topic_num=LDAP.num_topic, PLSA=LDAP.plsa, corpus_file=LDAP.corpus_file, corpus_percent=LDAP.corpus_percent, output_path=LDAP.output_path, mimic_movie_wiki=LDAP.mimic0_movie1_wiki2)
#     # lt.train_wiki_lda()
#     lt.train_lda(passes=5)
#     lt.save_phi_alpha_theta_topicdistrib()
#     # print lt.get_alpha()
#     # lt.get_topic_distrib_of_word()
#     pass
