import numpy as np
import sys
from os import path
import pandas as pd
import argparse
from gensim.models import Word2Vec, FastText


sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
from pre_models.build_wikipedia_data import get_filter_data


def process_embedding(docs_path, file_name, embed_type, vec_len, movie_formal=0, skip_gram=0, window=5):
    if movie_formal != 2:
        selected_docs = pd.read_csv(docs_path+file_name, header=None, index_col=[0]).values
        texts = [[word for word in doc[0].split(' ')] for doc in selected_docs]
    else:
        texts = get_filter_data(docs_path)
    if movie_formal == 0:
        name_append = ''
    elif movie_formal == 1:
        name_append = 'movie_review_'
    else:
        name_append = 'wiki_'
    if skip_gram == 0:
        sg = ''
    else:
        sg = 'skipgram'
    if embed_type == 0:
        embed_name = 'word2vec_'
        model = Word2Vec(texts, size=vec_len, window=window, min_count=0, workers=4, sg=skip_gram)
    else:
        embed_name = 'fasttext_'
        model = FastText(texts, size=vec_len, window=window, min_count=0, workers=4, sg=skip_gram)
    model.save(docs_path + name_append + embed_name + sg + str(vec_len) + '_window' + str(window) + '.model')


def get_embedding(doc_path, embed_type, vec_len, movie_formal=0, skip_gram=0, window=5):
    if movie_formal == 0:
        name_append = ''
    elif movie_formal == 1:
        name_append = 'movie_review_'
    else:
        name_append = 'wiki_'
    if skip_gram == 0:
        sg = ''
    else:
        sg = 'skipgram'
    if embed_type == 0:
        embed_name = 'word2vec_'
        return Word2Vec.load(doc_path + name_append + embed_name + sg + str(vec_len) + '_window' + str(window) + '.model')
    else:
        embed_name = 'fasttext_'
        return FastText.load(doc_path + name_append + embed_name + sg + str(vec_len) + '_window' + str(window) + '.model')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-pa", "--Path", type=str, help="the path of training corpus", default="")
    parser.add_argument("-cf", "--MovieRevieworFormal", type=int, help="0 for Formal MIMIC; 1 for Movie Review data and 2 for Wiki data", default=0)
    parser.add_argument("-em", "--Embedding", type=int, help="0 for Word2Vec; 1 for FastText", default=0)
    parser.add_argument("-sg", "--skip_gram", type=int, help="using skip-gram or not", default=0)
    parser.add_argument("-vl", "--vec_len", type=int, help="length of embedding vector ", default=20)
    parser.add_argument("-window", "--window", type=int, help="word window for WOB word2vec", default=5)
    args = parser.parse_args()
    if args.MovieRevieworFormal == 0:
        process_embedding(args.Path, 'selected_docs4LDA.csv', args.Embedding, args.vec_len, movie_formal=args.MovieRevieworFormal, skip_gram=args.skip_gram, window=args.window)
    elif args.MovieRevieworFormal == 1:
        process_embedding(args.Path, 'selected_movie_review_docs4LDA.csv', args.Embedding, args.vec_len, movie_formal=args.MovieRevieworFormal, skip_gram=args.skip_gram, window=args.window)
    else:
        process_embedding(args.Path, '', args.Embedding, args.vec_len, movie_formal=args.MovieRevieworFormal, skip_gram=args.skip_gram, window=args.window)

