# -*- coding: utf-8 -*-

import os
import pickle
import random
import argparse
import torch as t
import numpy as np
import time
import sys
from os import path
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from model import Word2Vec, SGNS
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])
# from utilities.csv_utility import CsvUtility
from lda import LdaTools
from params import LDAP, EMBEDP


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_negs', type=int, default=20, help="number of negative samples")
    parser.add_argument('--epoch', type=int, default=100, help="number of epochs")
    parser.add_argument('--mb', type=int, default=128, help="mini-batch size")
    parser.add_argument('--ss_t', type=float, default=1e-5, help="subsample threshold")
    parser.add_argument('--weights', action='store_true', help="use weights for negative sampling")
    parser.add_argument('--cuda', action='store_true', help="use CUDA")
    return parser.parse_args()


class PermutedSubsampledCorpus(Dataset):

    def __init__(self, datapath, ws=None):
        data = pickle.load(open(datapath, 'rb'))
        if ws is not None:
            self.data = []
            for iword, owords in data:
                if random.random() > ws[iword]:
                    self.data.append((iword, owords))
        else:
            self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        iword, owords = self.data[idx]
        return iword, np.array(owords)


def train(args):
    if LDAP.mimic0_movie1_wiki2 == 0:
        name = "MIMIC"
    elif LDAP.mimic0_movie1_wiki2 == 1:
        name = "MovieReview"
    else:
        name = "Wiki"
    idx2word = pickle.load(open(os.path.join(LDAP.output_path, name + '_idx2word.dat'), 'rb'))
    wc = pickle.load(open(os.path.join(LDAP.output_path, name + '_wc.dat'), 'rb'))
    wf = np.array([wc[word] for word in idx2word])
    wf = wf / wf.sum()
    ws = 1 - np.sqrt(args.ss_t / wf)
    ws = np.clip(ws, 0, 1)
    vocab_size = len(idx2word)
    weights = ws if args.weights else None

    model = Word2Vec(vocab_size=vocab_size, embedding_size=EMBEDP.veclen)
    time_code = '_#' + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '#'
    sgns = SGNS(embedding=model, vocab_size=vocab_size, n_negs=args.n_negs, weights=weights)
    if args.cuda:
        sgns = sgns.cuda()
    optim = Adam(sgns.parameters())
    test_data = pickle.load(open(os.path.join(LDAP.output_path, name + '_train.dat'), 'rb'))
    # for iword, oword in test_data:
    #     print(iword, type(iword))
    #     print(oword, type(oword))

    for epoch in range(1, args.epoch + 1):
        dataset = PermutedSubsampledCorpus(os.path.join(LDAP.output_path, name + '_train.dat'))
        dataloader = DataLoader(dataset, batch_size=args.mb, shuffle=True)
        total_batches = int(np.ceil(len(dataset) / args.mb))
        pbar = tqdm(dataloader)
        pbar.set_description("[Epoch {}]".format(epoch))
        for iword, owords in pbar:
            # print(iword.size(), owords.size())
            loss = sgns(iword, owords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())
    idx2vec = model.ivectors.weight.data.cpu().numpy()
    pickle.dump(idx2vec, open(os.path.join(LDAP.output_path, name + '_idx2vec.dat'), 'wb'))
    t.save(sgns.state_dict(), os.path.join(LDAP.output_path, '{}.pt'.format(name + '_model')))
    t.save(optim.state_dict(), os.path.join(LDAP.output_path, '{}.optim.pt'.format(name + '_model')))

if __name__ == '__main__':
    train(parse_args())
