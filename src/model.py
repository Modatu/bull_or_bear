# local
import model_utils

# global
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time

import sys

input_dim = 1
hidden_dim = 32
output_dim = 3
batch_size = 1
USE_CUDA = True


def fit_lstm(x, y, x_test, y_test, epochs):

    print_every = 2
    model = model_utils.LSTMNet_simple_mto(input_dim, hidden_dim, 1).cuda()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    train_size = len(y[0])
    cost = 0.

    for epoch in range(epochs):
        
        num_batches = train_size // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
    
            cost += model_utils.train_lstm(model, loss, optimizer, x[:, start:end, :], y[:, start:end, :])
        predY = model_utils.predict_lstm(model, x_test)
        predY = predY.flatten()

        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = cost / print_every
            cost = 0
            print_summary = 'time elapsed: %s (%d %d%%) ave. loss: %.4f' % (model_utils.time_since(start_time, float(epoch) / float(epochs)), epoch, float(epoch) / float(epochs) * 100, print_loss_avg)
            print(print_summary)
        
    return predY

def fit_enc_dec(x, y, x_test, y_test, epochs):

    print_every = 2
    model_encoder = model_utils.EncoderRNN(input_dim, hidden_dim, 1).cuda()
    model_decoder = model_utils.DecoderRNN(hidden_dim, output_dim).cuda()
    loss = nn.MSELoss()
    optimizer_encoder = optim.Adam(model_encoder.parameters(), lr=0.001)
    optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=0.001)
    start_time = time.time()
    train_size = len(y[0])

    print 'train_size', train_size
    cost = 0.

    for epoch in range(epochs):
        
        num_batches = train_size // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
    
            cost += model_utils.train_encoder_decoder(model_encoder, model_decoder, loss,
                                                      optimizer_encoder, optimizer_decoder,
                                                      x[:, start:end, :], y[:, start:end, :])
        
        predY = model_utils.predict_enc_dec(model_encoder, model_decoder, x_test)
        predY = predY.flatten()


        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = cost / print_every
            cost = 0
            print_summary = 'time elapsed: %s (%d %d%%) ave. loss: %.4f' % (model_utils.time_since(start_time, float(epoch) / float(epochs)), epoch, float(epoch) / float(epochs) * 100, print_loss_avg)
            print(print_summary)
        
    return predY


def fit_att_enc_dec(x, y, x_test, y_test, epochs):

    print_every = 2
    model_encoder = model_utils.EncoderRNN(input_dim, hidden_dim, 1).cuda()
    model_decoder = model_utils.AttentionDecoderRNN(hidden_dim, output_dim).cuda()
    # print(model_encoder)
    # print(model_decoder)
    loss = nn.MSELoss()
    optimizer_encoder = optim.Adam(model_encoder.parameters(), lr=0.001)
    optimizer_decoder = optim.Adam(model_decoder.parameters(), lr=0.001)
    start_time = time.time()
    train_size = len(y[0])
    test_size = len(y_test)
    predY = np.zeros(test_size)

    print 'train_size', train_size
    cost = 0.

    for epoch in range(epochs):
        
        num_batches = train_size // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size

            cost_temp = model_utils.train_att_encoder_decoder(model_encoder, model_decoder, loss,
                                                      optimizer_encoder, optimizer_decoder,
                                                      x[:, start:end, :], y[:, start:end, :])
            cost += cost_temp



        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = cost / print_every
            cost = 0
            print_summary = 'time elapsed: %s (%d %d%%) ave. loss: %.4f' % (model_utils.time_since(start_time, float(epoch) / float(epochs)), epoch, float(epoch) / float(epochs) * 100, print_loss_avg)
            print(print_summary)

    num_batches = test_size // batch_size
    for k in range(num_batches):
        start, end = k * batch_size, (k + 1) * batch_size

        predY[k] = model_utils.predict_enc_attdec(model_encoder, model_decoder, x_test[:, start:end, :])

    return predY