import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import numpy as np
import sys
from os import path
import time
sys.path.append(path.split(path.abspath(path.dirname(__file__)))[0])

from params import LSTMP
from neural_networks.lstm_define import Net
from com.test import test
from utilities.metirc_utility import get_accuracy, get_accuracy_gpu

def train(train_x, train_y, test_x, test_y):
    train_size = train_x.shape[0]
    input_size = train_x.shape[-1]

    train_dataset = Data.TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=LSTMP.batchsize, shuffle=True, num_workers=2)

    net = Net(input_size, LSTMP.hidden_size, LSTMP.num_classes, LSTMP.num_layers, LSTMP.drop_out, LSTMP.use_gpu)
    if LSTMP.use_gpu:
        net.cuda()

    # Loss and Optimizer
    criterion = nn.BCELoss(size_average=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=LSTMP.learning_rate, weight_decay=LSTMP.weight_decay)

    result_epoch = []
    time_epoch = []
    for epoch in range(LSTMP.num_epochs):
        start_time = time.time()
        for i, data_iter in enumerate(train_loader, 0):
            data_x, data_y = data_iter
            TX = Variable(data_x).float()
            TY = Variable(data_y).float()
            if LSTMP.use_gpu:
                TX = TX.cuda()
                TY = TY.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()
            output = net(TX)

            if LSTMP.use_gpu:
                acc_list, acc = get_accuracy_gpu(TY.data, output.data)
            else:
                acc_list, acc = get_accuracy(TY.data.numpy(), output.data.numpy())
            mlp_loss = criterion(output, TY)
            mlp_loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 1:
                print 'Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Acc: %.4f' \
                      % (epoch + 1, LSTMP.num_epochs, i + 1, train_size / LSTMP.batchsize, mlp_loss.data[0], acc)
                # test model
        consume_time = time.time() - start_time
        time_epoch.append(consume_time)
        auc_list, accuracy_mean, precision_mean, recall_mean, f1_mean = test(test_x, test_y, net, LSTMP)
        print 'Epoch [%d/%d], AUC:' % (epoch + 1, LSTMP.num_epochs), auc_list[:4], 'ACC:', accuracy_mean, 'Precision:', precision_mean, 'Recall:', recall_mean, 'F1:', f1_mean
        result_epoch.append([epoch + 1, auc_list, accuracy_mean, precision_mean, recall_mean, f1_mean])

    return net, [], result_epoch, np.array(time_epoch)
