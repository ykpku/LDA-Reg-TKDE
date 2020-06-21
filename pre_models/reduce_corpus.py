import pandas as pd
import numpy as np
from gensim import corpora
import sys
from os import path

sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])


def reduce_corpus():
    pass


if __name__ == '__main__':
    # get_good_docs('../data-repository/result/jack_1.csv', 10, 2)
    # file_list = Directory.folder_process(Path+'/data-repository/CMS_result')
    #
    # merge_dict = dict({})
    # doc_map = []
    # for file_path in file_list:
    #     dict_tmp = get_good_docs(file_path, 10, 5)
    #     print 'this dict len : ', len(dict_tmp)
    #     merge_dict.update(dict_tmp)
    #     print 'after the merge : ', len(merge_dict)
    #     doc_map.extend(get_docs_frequence_kind_map(file_path=file_path))
    # draw_pl(x_y=doc_map, type='o')
    # print len(merge_dict)
    # texts = [[word for word in doc.split(' ')] for doc in merge_dict.values()]
    # # pprint(texts[:5])
    # dictionary = corpora.Dictionary(texts)
    # # dictionary.save(Path+'/data-repository/available_word_in_literature.dict')
    # print dictionary
    #
    # CsvUtility.write_dict2csv(merge_dict, Path+'/data-repository', 'selected_CMS_docs4LDA.csv')
    pass

