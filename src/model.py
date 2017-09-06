# local
import model_utils

# global
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time

input_dim = 1
hidden_dim = 32
output_dim = 1
batch_size = 1


def fit(x, y, x_test, y_test, epochs):

    print_every = 2
    model = model_utils.LSTMNet_simple_mto(input_dim, hidden_dim, 1).cuda()
    loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    start_time = time.time()
    train_size = len(y[0])
    cost = 0.

    # has_init_hidden = getattr(model, 'initHidden', None)
    # if has_init_hidden:
    #     print('initialise hidden states')
    #     model.initHidden()
    

    for epoch in range(epochs):
        
        num_batches = train_size // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
    
            cost += model_utils.train(model, loss, optimizer, x[:, start:end, :], y[:, start:end, :])
        predY = model_utils.predict(model, x_test)
        predY = predY.flatten()
        # print np.mean(np.sqrt(predY**2 - y_test**2))
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
    cost = 0.

    # has_init_hidden = getattr(model, 'initHidden', None)
    # if has_init_hidden:
    #     print('initialise hidden states')
    #     model.initHidden()
    

    for epoch in range(epochs):
        
        num_batches = train_size // batch_size
        for k in range(num_batches):
            start, end = k * batch_size, (k + 1) * batch_size
    
            cost += model_utils.train_encoder_decoder(model_encoder, model_decoder, loss,
                                                      optimizer_encoder, optimizer_decoder,
                                                      x[:, start:end, :], y[:, start:end, :])
        # predY = model_utils.predict(model, x_test)
        # predY = predY.flatten()
        # print np.mean(np.sqrt(predY**2 - y_test**2))
        predY = 0
        if epoch == 0: continue

        if epoch % print_every == 0:
            print_loss_avg = cost / print_every
            cost = 0
            print_summary = 'time elapsed: %s (%d %d%%) ave. loss: %.4f' % (model_utils.time_since(start_time, float(epoch) / float(epochs)), epoch, float(epoch) / float(epochs) * 100, print_loss_avg)
            print(print_summary)
        
    return predY