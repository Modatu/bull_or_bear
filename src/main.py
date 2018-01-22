# main routine here the real stuff gets done

# imports
# local
import model_utils
import model

# global
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import numpy as np

import matplotlib.pyplot as plt

import sys

torch.manual_seed(42)


# load dataset
x_np = np.load('/home/florian/src/bull_or_bear/data/x_train.npy')
y_np = np.load('/home/florian/src/bull_or_bear/data/y_train.npy')
x_np_test = np.load('/home/florian/src/bull_or_bear/data/x_test.npy')
y_np_test = np.load('/home/florian/src/bull_or_bear/data/y_test.npy')

x = Variable(torch.cuda.FloatTensor(x_np), requires_grad=False)
y = Variable(torch.cuda.FloatTensor(y_np), requires_grad=False)
x_test = Variable(torch.cuda.FloatTensor(x_np_test), requires_grad=False)
y_test = y_np_test

print 'boo', 4 % 3
# The interesting part begins
# actual training of model
epochs = 16
# model_type = 'lstm'
model_type = 'seq2seq'
attention = False

print 'model type: ', model_type
print 'attention: ' , attention
if model_type == 'seq2seq':
  if attention:
    predY = model.fit_att_enc_dec(x, y, x_test, y_test, epochs)
  else:
    predY = model.fit_enc_dec(x, y, x_test, y_test, epochs)
else:
  predY = model.fit_lstm(x, y, x_test, y_test, epochs)

# print np.column_stack((predY, y_test))

pred,  = plt.plot(predY, ls='--', linewidth=2, label='predictions', color='tomato')
truth,  = plt.plot(y_test, linewidth=2, label='ground truth', color='indigo')
# plt.legend(handles=[pred, truth])
plt.legend(bbox_to_anchor=(0.7, 0.95), loc=2, borderaxespad=0.)
plt.show()