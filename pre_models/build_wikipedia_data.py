import re
import sys
from os import path
import os
import glob
import nltk.tokenize
import json
from bs4 import BeautifulSoup
import numpy as np
from multiprocessing import Pool
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from utilities.csv_utility import CsvUtility
from utilities.document_utility import Document2VecUtility



def _process_text(sentences, word2index, remove_stopwords=True, stem_words=False, remove_html=False):
    # Remove HTML
    if remove_html:
        sentences = BeautifulSoup(sentences, "lxml").get_text()
        pass

    # Remove non-letters
        sentences = re.sub("[^a-zA-Z]", " ", sentences)

    # Convert words to lower case and split them
    words = sentences.lower().split()

    # Optionally remove stop words (false by default)
    if remove_stopwords:

        stops = set([])
        file_stop = open('en.txt')
        st_line = file_stop.readline()
        while st_line:
            stops.add(st_line.strip())
            st_line = file_stop.readline()
        words = [w for w in words if not w in stops]
    if stem_words:
        words = [nltk.PorterStemmer().stem(w) for w in words]

    real_text = []
    word_kind = set([])
    for w in words:
        if w in word2index:
            real_text.append(w)
            if w not in word_kind:
                word_kind.add(w)
    return real_text, len(word_kind)

def get_real_word_list(file_name, word2index, word_kind_limit,remove_stopwords=False, stem_words=False, remove_html=False):
    with open(file_name, "r") as f:
        all_pages = []

        # Read all json objects in the file
        for x in f.readlines():

            x = json.loads(x.strip())
            x_text, word_kind = _process_text(x["text"], word2index, remove_stopwords, stem_words, remove_html)
            # print("-------------")
            # print(x["text"])
            # print(x_text)
            # print(word_kind)
            # print("-------------")
            # Enumerate every sentence in the body text
            if word_kind >= word_kind_limit:
                all_pages.append(x_text)

    return all_pages


def get_corpus_contend(file_list, word2index, word_kind_limit=50, remove_stopwords=False, stem_words=True, remove_html=True):
    print('file list size : ', len(file_list))

    corpus_contend = []
    for file_iter, file_name in enumerate(file_list):
        tem_data = get_real_word_list(file_name, word2index, word_kind_limit, remove_stopwords, stem_words, remove_html)
        print(file_name, 'read ready~', len(tem_data))

        corpus_contend.extend(tem_data)
        if file_iter % 1000 == 0:
            print(file_iter, 'file done.')
    print('file number', len(corpus_contend))
    return corpus_contend

def get_corpus_contend_thread(process_index, file_list, word2index, write_path="/home1/yk/wikipedia_dataset/filter", word_kind_limit=50, remove_stopwords=False, stem_words=True, remove_html=True):

    corpus_contend = []
    for file_iter, file_name in enumerate(file_list):
        tem_data = get_real_word_list(file_name, word2index, word_kind_limit, remove_stopwords, stem_words, remove_html)
        # print(file_name, 'read ready~', len(tem_data))

        corpus_contend.extend(tem_data)
        if (file_iter+1) % 10 == 0:
            print((file_iter + 1), 'file done.')
            if (file_iter+1) % 100 == 0:
                name = process_index + "process_" + str(file_iter+1) + "iter_text.csv"
                CsvUtility.write_norm_array2csv(corpus_contend, write_path, name)
                corpus_contend = []

    print(process_index, 'finish~')
    return corpus_contend

def _load_and_process_metadata(sentence_dir, movie_review_path, num_processor=8):

    # Extract the filenames.
    sentence_filenames = glob.glob(os.path.join(sentence_dir, "*/*"))
    print(len(sentence_filenames))
    # print(sentence_filenames[-10:])
    # sentence_filenames = sentence_filenames[-20:]

    word2index = CsvUtility.read_pickle(movie_review_path + '/new_word2index.pkl', 'r')
    index2word = CsvUtility.read_pickle(movie_review_path + '/new_index2word.pkl', 'r')

    # Break the files into num_threads batches.
    spacing = np.linspace(0, len(sentence_filenames), num_processor + 1).astype(np.int)
    ranges = []
    for i in xrange(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    p = Pool(num_processor)
    res = []
    for i in range(num_processor):
        start = ranges[i][0]
        end = ranges[i][1]
        res.append(p.apply_async(get_corpus_contend_thread, args=(str(i), sentence_filenames[start:end], word2index)))
        print(str(i) + ' processor started !')

    # get_corpus_contend(sentence_filenames, word2index)
    p.close()
    p.join()
    # filter_contend = {}
    # filter_index = 0
    # for i in res:
    #     for a in i.get():
    #         filter_contend[str(filter_index)] = ' '.join(a)
    #         filter_index += 1
    # CsvUtility.write_dict2csv(filter_contend, sentence_dir, 'selected_movie_review_docs4LDA.csv')

def get_filter_data(path):
    get_con = []
    for process_index in range(1):
        for file_iter in range(6):
            name = str(process_index) + "process_" + str(file_iter + 1) + "00iter_text.csv"
            content = CsvUtility.read_norm_array_csv(path, name)
            # print(len(content))
            get_con.extend(content)
    print(" content number : ", len(get_con))
    return get_con[:100000]
            # print(content[0])


if __name__ == '__main__':
    # _load_and_process_metadata("/home1/yk/wikipedia_dataset/text", "/home1/yk/Movie_Review_data", num_processor=20)
    contend = get_filter_data("/home1/yk/wikipedia_dataset/filter")
    name = "wiki_text.csv"
    CsvUtility.write_norm_array2csv(contend, "/home1/yk/wikipedia_dataset/filter", name)
    pass
