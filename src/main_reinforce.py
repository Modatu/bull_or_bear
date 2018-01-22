# main routine here the real stuff gets done
from __future__ import print_function

# imports
# local
import model_utils
import model
import reinforce_utils

# global
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

import sys

from collections import namedtuple



# script starts

# load data
x_np = np.load('../data/x_train.npy')
y_np = np.load('../data/y_train.npy')
x_np_test = np.load('../data/x_test.npy')
y_np_test = np.load('../data/y_test.npy')
x_np_ref = np.load('../data/x_ref.npy')
y_np_ref = np.load('../data/y_ref.npy')
print(x_np_ref.shape)
print(x_np.shape)
print(y_np.shape)

Options = namedtuple('Options', 'credit, capacity, stock, ind, input_dim, hidden_dim, output_dim, window_size, batch_size, nlayer, lr, steps, gamma')
Options_really_now = Options(credit = 10000,
        capacity = 6000,
        stock = 0,
        ind = 0,
        input_dim = 6,
        hidden_dim = 64,
        output_dim = 3,
        window_size = 90,
        nlayer = 2,
        batch_size = 8,
        lr = 0.0028,
        steps = 10000000,
        gamma = 0.8)

print(Options_really_now.credit)
print('len', len(x_np[1]))
env = reinforce_utils.Env(x_np, y_np, x_np_ref, y_np_ref, Options_really_now.credit, Options_really_now.stock, Options_really_now.ind)
# model = model_utils.LSTM_DQN(Options_really_now.input_dim, Options_really_now.hidden_dim, Options_really_now.output_dim).cuda()
# target_model = model_utils.LSTM_DQN(Options_really_now.input_dim, Options_really_now.hidden_dim, Options_really_now.output_dim).cuda()
model = model_utils.LSTM_DDDQN(Options_really_now.input_dim, Options_really_now.hidden_dim, Options_really_now.output_dim).cuda()
target_model = model_utils.LSTM_DDDQN(Options_really_now.input_dim, Options_really_now.hidden_dim, Options_really_now.output_dim).cuda()
memory = reinforce_utils.Memory(Options_really_now)


agent = reinforce_utils.DDQNAgent(Options_really_now, env, model, target_model, memory)

agent.fit()
print('credit (start))', env.start_credit)
print('credit (end))', env.credit)

