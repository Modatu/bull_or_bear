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


def windower(data_x, data_y, windowsize):
  array_x = np.zeros((len(data_x)-windowsize-10, windowsize))
  # many to one
  array_y = np.zeros((len(data_x)-windowsize-10, 1))
  # many to many
  # array_y = np.zeros((len(data_x)-windowsize-10, windowsize))
  print len(data_x)
  for i in range(len(data_x)-windowsize-10):
    array_x[i] = data_x[i:windowsize+i]
    # many to one
    array_y[i] = data_y[i+windowsize+1:i+windowsize+2]
    # many to many
    # array_y[i] = data_y[i+windowsize+1:i+(2*windowsize)+1]
  return array_x, array_y

torch.manual_seed(42)


# generate training data, lame shit
seq_length = 50
x_np = np.linspace(0, 100*np.pi, 2000)
y_np = np.sin(x_np)
x_np = y_np
half = int(len(x_np)/2)
x_np_train = x_np[0:half-1]
x_np_test = x_np[half:-1]
y_np_train = y_np[0:half-1]
y_np_test = y_np[half:-1]
x_np, y_np = windower(x_np_train, y_np_train, seq_length)
x_np_test, y_np_test = windower(x_np_test, y_np_test, seq_length)
x_np = np.reshape(x_np, (x_np.shape[0],x_np.shape[1],1))
y_np = np.reshape(y_np, (y_np.shape[0],y_np.shape[1],1))
x_np_test = np.reshape(x_np_test, (x_np_test.shape[0],x_np_test.shape[1],1))
y_np_test = np.reshape(y_np_test, (y_np_test.shape[0],y_np_test.shape[1],1))
# Convert to the shape (seq_length, num_samples, input_dim)
x_np = np.swapaxes(x_np, 0, 1)
y_np = np.swapaxes(y_np, 0, 1)
x_np_test = np.swapaxes(x_np_test, 0, 1)
y_np_test = y_np_test.flatten()
x = Variable(torch.cuda.FloatTensor(x_np), requires_grad=False)
y = Variable(torch.cuda.FloatTensor(y_np), requires_grad=False)
x_test = Variable(torch.cuda.FloatTensor(x_np_test), requires_grad=False)
y_test = y_np_test
print 'shape x:', x_np.shape
print 'shape y', y_np.shape

# The interesting part begins
# actual training of model
epochs = 8
model_type = 'lstm'
model_type = 'seq2seq'

print 'model type: ', model_type
if model_type == 'seq2seq':
  predY = model.fit_enc_dec(x, y, x_test, y_test, epochs)
else:
  predY = model.fit_lstm(x, y, x_test, y_test, epochs)

print np.column_stack((predY, y_test))

plt.plot(predY, ls='--', linewidth=2, color='tomato')
plt.plot(y_test, linewidth=2, color='indigo')
plt.show()