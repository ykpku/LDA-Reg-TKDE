# -*- coding=utf-8 -*-
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from numpy import linalg as LA
import numpy as np
import sys
from os import path
import time
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from params import MLPP, ldaregP, LDAP
from neural_networks.mlp_hook_define import Net
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

def train(train_x, train_y, lda_model, model_path):
    if train_x.ndim == 3:
        train_x = np.sum(train_x, axis=1, keepdims=False)
    train_size = train_x.shape[0]
    input_size = train_x.shape[-1]
    train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=MLPP.batchsize, shuffle=True, num_workers=2)

    net = torch.load(model_path)

    # define new mlp for better hook
    # net = Net(input_size, MLPP.hidden_size, MLPP.num_classes, MLPP.num_layers, sparse=MLPP.sparse_update)

    # -------------- find gradient with hook--------------------
    # total_grad_out = []
    # total_grad_in = []
    # def hook_fn_backward(module, grad_input, grad_output):
    #     print(module)  # 为了区分模块
    #     # 为了符合反向传播的顺序，我们先打印 grad_output
    #     print('grad_output', grad_output)
    #     # 再打印 grad_input
    #     print('grad_input', grad_input)
    #     # 保存到全局变量
    #     total_grad_in.append(grad_input)
    #     total_grad_out.append(grad_output)
    # modules = net.named_children()
    # for name, module in modules:
    #     module.register_backward_hook(hook_fn_backward)
    # -------------- find gradient with hook--------------------

    if MLPP.use_gpu:
        net = net.cuda()

    # Loss and Optimizer
    criterion = nn.BCELoss(size_average=True)

    def input_reltated(x):
        if len(x.size()) == 2:
            return input_size in x.size()
        return False

    def other_reltated(x):
        if len(x.size()) == 2:
            return not (input_size in x.size())
        return True

    input_related_param = filter(input_reltated, net.parameters())
    other_param = filter(other_reltated, net.parameters())
    optimizer = torch.optim.Adam([{'params': input_related_param, 'weight_decay': 0.0}, {'params': other_param, 'weight_decay': MLPP.weight_decay}], lr=MLPP.learning_rate)

    alpha, phi = lda_model.read_phi_alpha_theta()
    phi_kw = torch.from_numpy(phi).float().cuda()
    alpha = (alpha * ldaregP.param_alpha + 1)
    alpha = torch.from_numpy(alpha.flatten()).float().cuda()
    sita_dk = np.ones((MLPP.hidden_size, lda_model.topic_num)) * (1.0 / lda_model.topic_num)
    sita_dk = torch.from_numpy(sita_dk).float().cuda()

    neuron_gradient = []
    for epoch in range(MLPP.num_epochs):
        for i, data_iter in enumerate(train_loader, 0):

            # Convert numpy array to torch Variable
            data_x, data_y = data_iter
            TX = Variable(data_x).float()
            TY = Variable(data_y).float()
            if MLPP.use_gpu:
                TX = TX.cuda()
                TY = TY.cuda()

            # Forward + Backward
            optimizer.zero_grad()
            output = net(TX)
            mlp_loss = criterion(output, TY)
            mlp_loss.backward()

            # lda_reg operation
            param_list = list(net.parameters())
            for f_i, f in enumerate(param_list):
                if f_i == 0:
                    r_DKN = calcResponsibility(sita_dk, phi_kw)
                    LDA_gradient = calcRegGrad(sita_DK=sita_dk, phi_KW=phi_kw, weight=f.data)
                    reg_grad_w = LDA_gradient / train_size / MLPP.num_classes
                    f.grad.data = f.grad.data + reg_grad_w * ldaregP.param_lda
                    sita_dk = update_LDA_EM(alpha=alpha, responsibility_all_doc=r_DKN, weight=f.data, param_lda=ldaregP.param_lda)

            # x1_gradient
            x1_gradient = (output * (1 - output)).t() * param_list[2].data
            neuron_gradient.append(x1_gradient.data.cpu().numpy().tolist())

            optimizer.step()

            # print training accuracy
            # if MLPP.use_gpu:
            #     acc_list, acc = get_accuracy_gpu(TY.data, output.data)
            # else:
            #     acc_list, acc = get_accuracy(TY.data.numpy(), output.data.numpy())
            if (i + 1) % 100 == 1:
                print 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f' % (epoch + 1, MLPP.num_epochs, i + 1, train_size / MLPP.batchsize, mlp_loss.data[0])

            # hooker save the result
            # print('==========Saved inputs and outputs==========')
            # for idx in range(len(total_grad_in)):
            #     print('grad output: ', total_grad_out[idx].size())
            #     print('grad input: ', total_grad_in[idx].size())
            # total_grad_out = []
            # total_grad_in = []
            # print(i)

    return net, sita_dk.cpu().numpy(), neuron_gradient

