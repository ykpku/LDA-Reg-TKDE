import sys
import time
from os import path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable

sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from params import ldaregP, LDAP, LSTMP
from neural_networks.lstm_define import Net
from com.test import test
from utilities.metric_utility import get_accuracy, get_accuracy_gpu

def calcResponsibility(theta_alldoc, phi_KW):
    phi = torch.t(phi_KW)
    doc_num = theta_alldoc.size()[0]
    topic_num = theta_alldoc.size()[1]
    word_num = phi.size()[0]

    theta_alldoc = theta_alldoc.view(doc_num, 1, topic_num)
    phi = phi.contiguous().view(1, word_num, topic_num)
    responsibility_all_doc = torch.mul(theta_alldoc, phi)
    sum_r_DW = torch.sum(responsibility_all_doc, 2)
    zero_idx = (sum_r_DW == 0)
    responsibility_all_doc.masked_fill_(zero_idx.view(doc_num, word_num, 1), 1. / theta_alldoc.size()[1])
    responsibility_all_doc = responsibility_all_doc / torch.sum(responsibility_all_doc, 2, keepdim=True)
    return responsibility_all_doc


def calcRegGrad(sita_DK, phi_KW, weight):
    theta_phi = sita_DK.mm(phi_KW)
    zero_idx = (theta_phi == 0)
    theta_phi = torch.log(theta_phi)
    theta_phi[zero_idx] = -100

    re = -1.0 * torch.sign(weight) * theta_phi
    return re

def update_LDA_EM(alpha, responsibility_all_doc, weight, param_lda):
    doc_num = responsibility_all_doc.size()[0]
    word_num = responsibility_all_doc.size()[1]
    topic_num = responsibility_all_doc.size()[2]
    weight_DWK = torch.abs(param_lda * weight).view(doc_num, word_num, 1)
    new_sita = torch.sum(responsibility_all_doc.mul(weight_DWK), 1) + (alpha - 1).view(1, topic_num)
    new_sita = new_sita / torch.sum(new_sita, 1, keepdim=True)
    return new_sita

def train(train_x, train_y, test_x, test_y, lda_model):
    train_size = train_x.shape[0]
    input_size = train_x.shape[-1]

    train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=LSTMP.batchsize, shuffle=True, num_workers=2)

    net = Net(input_size, LSTMP.hidden_size, LSTMP.num_classes, LSTMP.num_layers, LSTMP.drop_out, LSTMP.use_gpu)
    if LSTMP.use_gpu:
        net.cuda()

    # Loss and Optimizer
    criterion = nn.BCELoss(size_average=True)

    def input_reltated(x):
        if len(x.size()) == 2:
            return input_size in x.size()
        return False

    def other_reltated(x):
        if len(x.size()) == 2:
            return not(input_size in x.size())
        return True

    input_related_param = filter(input_reltated, net.parameters())
    other_param = filter(other_reltated, net.parameters())
    optimizer = torch.optim.Adam([{'params': input_related_param, 'weight_decay': 0.0}, {'params': other_param, 'weight_decay': LSTMP.weight_decay}], lr=LSTMP.learning_rate)

    # lda result
    alpha, phi = lda_model.read_phi_alpha_theta()
    phi_kw = torch.from_numpy(phi).float().cuda()
    # alpha learned from lda is less than 1, and lead to nan in regularization gradient
    alpha = (alpha * ldaregP.param_alpha + 1)
    alpha = torch.from_numpy(alpha.flatten()).float().cuda()
    sita_dk = np.ones((LSTMP.hidden_size * 4, LDAP.num_topic)) * (1.0/LDAP.num_topic)
    sita_dk = torch.from_numpy(sita_dk).float().cuda()

    result_epoch = []
    LDA_gradient = 0
    r_DKN = 0
    time_epoch = []
    for epoch in range(LSTMP.num_epochs):
        start_time = time.time()
        for i, data_iter in enumerate(train_loader):

            # Convert numpy array to torch Variable
            data_x, data_y = data_iter
            TX = Variable(data_x).float()
            TY = Variable(data_y).float()
            if LSTMP.use_gpu:
                TX = TX.cuda()
                TY = TY.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output = net(TX)
            lstm_loss = criterion(output, TY)
            lstm_loss.backward()

            for f_i, f in enumerate(list(net.parameters())):
                if f_i == 0:
                    if epoch < 2 or (i + 1) % ldaregP.paramuptfreq == 0:
                        r_DKN = calcResponsibility(sita_dk, phi_kw)
                        LDA_gradient = calcRegGrad(sita_DK=sita_dk, phi_KW=phi_kw, weight=f.data)

                    reg_grad_w = LDA_gradient / train_size / LSTMP.num_classes
                    f.grad.data = f.grad.data + reg_grad_w * ldaregP.param_lda

                    if epoch < 2 or (i + 1) % ldaregP.ldauptfreq == 0:
                        if epoch >= 2 and (i + 1) % ldaregP.paramuptfreq != 0:
                            r_DKN = calcResponsibility(sita_dk, phi_kw)
                        sita_dk = update_LDA_EM(alpha=alpha, responsibility_all_doc=r_DKN, weight=f.data, param_lda=ldaregP.param_lda)
            optimizer.step()
            if LSTMP.use_gpu:
                acc_list, acc = get_accuracy_gpu(TY.data, output.data)
            else:
                acc_list, acc = get_accuracy(TY.data.numpy(), output.data.numpy())
            if (i + 1) % 100 == 1:
                print 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f' % (epoch + 1, LSTMP.num_epochs, i + 1, train_size / LSTMP.batchsize, lstm_loss.data[0], acc)
            # test model
        consume_time = time.time() - start_time
        time_epoch.append(consume_time)
        auc_list, accuracy_mean, precision_mean, recall_mean, f1_mean = test(test_x, test_y, net, LSTMP)
        print 'Epoch [%d/%d], AUC:' % (epoch + 1, LSTMP.num_epochs), auc_list[:4], 'ACC:', accuracy_mean, 'Precision:', precision_mean, 'Recall:', recall_mean, 'F1:', f1_mean
        result_epoch.append([epoch+1, auc_list, accuracy_mean, precision_mean, recall_mean, f1_mean])

    return net, sita_dk.cpu().numpy(), result_epoch, np.array(time_epoch)

