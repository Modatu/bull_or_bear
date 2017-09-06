# utility functions and classes for model.py

# imports
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import math


#_______________code starts here_______________

use_cuda = True

# simple LSTMnet class for many to one mapping
class LSTMNet_simple_mto(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMNet_simple_mto, self).__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        out = x
        for i in range(self.n_layers):
        	out, _ = self.lstm.forward(out, (h0, c0))

        return self.linear.forward(out[-1])

 # simple LSTMnet class for many to many mapping
class LSTMNet_simple_mtm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMNet_simple_mtm, self).__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        out = x
        for i in range(self.n_layers):
        	out, _ = self.lstm.forward(out, (h0, c0))

        return self.linear.forward(out)

class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim)

    def forward(self, input_data, hidden_states):
        output = input_data
        for i in range(self.n_layers):
            output, hidden_states = self.lstm(output, hidden_states)
        return output, hidden_states

    def initHidden(self):
        h0 = Variable(torch.zeros(1, 1, self.hidden_dim))
        c0 = Variable(torch.zeros(1, 1, self.hidden_dim))
        if use_cuda:
          h0 = h0.cuda()
          c0 = c0.cuda()
          return (h0, c0)
        else:
          return (h0, c0)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.lstm =nn.LSTM(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_data, hidden_states):
        output = input_data
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden_states = self.lstm(output, hidden_states)
        output = self.linear(output[0])
        return output, hidden_states

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

# simple training function
def train_lstm(model, loss, optimizer, x,y):

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x, hidden)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]

def train_encoder_decoder(encoder, decoder, loss, optimizer_encoder, optimizer_decoder, x,y):

    # Reset gradient
    optimizer_encoder.zero_grad()
    # optimizer_decoder.zero_grad()

    # forward pass through encoder
    encoder_hidden = encoder.initHidden()
    encoder_outputs, encoder_hidden = encoder.forward(x, encoder_hidden)

    # set decoder hidden states equal to the encoder output hidden state
    decoder_hidden = encoder_hidden

    # forward pass through decoder
    output, hidden_states = decoder.forward(encoder_outputs, decoder_hidden)
    
    # Forward
    loss = loss.forward(output, y)

    # Backward
    loss.backward()

    # Update parameters
    optimizer_encoder.step()
    optimizer_decoder.step()

    return loss.data[0]


# simple predict function
def predict(model, x_test):
    # x = Variable(x_val, requires_grad=False)
    output = model.forward(x_test)
    return output.data.cpu().numpy()

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))