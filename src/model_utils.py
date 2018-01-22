# utility functions and classes for model.py

# imports global
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import time
import numpy as np
import sys
import random


#_______________code starts here_______________

USE_CUDA = True

# simple LSTMnet class for many to one mapping
class LSTMNet_simple_mto(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMNet_simple_mto, self).__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
    
    # not stateful
    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        out = x
        for i in range(self.n_layers):
        	out, _ = self.lstm.forward(out, (h0, c0))

        return self.linear.forward(out[-1])

class LSTM_DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTM_DQN, self).__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    # not stateful
    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(self.n_layers, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(self.n_layers, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        out = x
        for i in range(self.n_layers):
            out, _ = self.lstm.forward(out, (h0, c0))

        return self.linear.forward(out[-1])

class LSTM_DDDQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTM_DDDQN, self).__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, n_layers)
        self.val_ind = torch.LongTensor(self.output_dim).fill_(0).unsqueeze(0)
        self.adv_ind = torch.LongTensor(np.arange(1, self.output_dim + 1)).unsqueeze(0)
        # self.lstm_val = nn.LSTM(hidden_dim, 1, n_layers)
        # self.lstm_adv = nn.LSTM(hidden_dim, self.output_dim)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim +1 , bias=False)
        
    # not stateful
    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        val = Variable(self.val_ind).cuda()
        adv = Variable(self.adv_ind).cuda()
        out = x
        for i in range(self.n_layers):
            out, _ = self.lstm.forward(out, (h0, c0))
        
        out = self.linear(out[-1])
        value = out.gather(1, val.expand(out.size(0), self.output_dim))
        advantage = out.gather(1, adv.expand(out.size(0), self.output_dim))
        q = value + (advantage - advantage.mean(1, keepdim=True))

        return q

 # simple LSTMnet class for many to many mapping
class LSTMNet_simple_mtm(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=1):
        super(LSTMNet_simple_mtm, self).__init__()
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)
    
    # not stateful
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

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers)

    def forward(self, input_data, hidden_states):
        output = input_data
        for i in range(self.n_layers):
            output, hidden_states = self.lstm(output, hidden_states)
        return output, hidden_states

    # not stateful
    def initHidden(self, batch_size):
        h0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        c0 = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if USE_CUDA:
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

        self.lstm =nn.LSTM(hidden_dim, hidden_dim, num_layers=n_layers)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_data, hidden_states):
        output = input_data
        for i in range(self.n_layers):
            output = F.relu(output)
            output, hidden_states = self.lstm(output, hidden_states)
        output = self.linear(output[0])
        return output, hidden_states


class AttentionDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, n_layers=1, dropout_p=0.1):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers     
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn(hidden_dim)
        self.lstm = nn.LSTM(hidden_dim*2, hidden_dim, num_layers=n_layers, dropout=dropout_p)
        self.linear = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, input_data, last_context, last_hidden, encoder_outputs):
        
        # Combine embedded input word and last context, run through RNN
        rnn_input = torch.cat((input_data, last_context), 1).unsqueeze(0)
        rnn_output, hidden = self.lstm(rnn_input, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = self.linear(torch.cat((rnn_output, context), 1))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.attn = nn.Linear(self.hidden_size, hidden_size)


    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len)) # B x 1 x S
        if USE_CUDA: attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])

        # Normalize energies to weights in range 0 to 1, resize to 1 x 1 x seq_len
        return F.softmax(attn_energies).unsqueeze(0).unsqueeze(0)
    
    def score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        energy = torch.dot(hidden.view(-1), energy.view(-1))
        return energy

# simple training function
def train_lstm(model, loss, optimizer, x,y):

    # Reset gradient
    optimizer.zero_grad()

    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward()

    # Update parameters
    optimizer.step()

    return output.data[0]


def train_encoder_decoder(encoder, decoder, loss, optimizer_encoder, optimizer_decoder, x,y):

    batch_size = x.size()[1]
    # Reset gradient
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()

    # forward pass through encoder
    encoder_hidden = encoder.initHidden(batch_size)
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


def train_att_encoder_decoder(encoder, decoder, loss, optimizer_encoder, optimizer_decoder, x,y):

    batch_size = x.size()[1]
    y_length = y.size()[0]

    # Reset gradient
    optimizer_encoder.zero_grad()
    optimizer_decoder.zero_grad()

    # forward pass through encoder
    encoder_hidden = encoder.initHidden(batch_size)
    encoder_outputs, encoder_hidden = encoder.forward(x, encoder_hidden)

    # prepare the rest of the decoder inputs
    decoder_input = Variable(torch.zeros(1, decoder.hidden_dim)).cuda()
    decoder_context = Variable(torch.zeros(1, decoder.hidden_dim)).cuda()
    # set decoder hidden states equal to the encoder output hidden state
    decoder_hidden = encoder_hidden

    use_teacher_forcing = random.random() < 0.5
    if use_teacher_forcing:
        for di in range(y_length):
            # forward pass through decoder
            output, context, hidden_states, attn_weights = decoder.forward(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

            # Forward
            loss = loss.forward(output, y[di])
            decoder_input = y[di]
    else:
        for di in range(y_length):
            # forward pass through decoder
            output, context, hidden_states, attn_weights = decoder.forward(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

            # Forward
            loss = loss.forward(output, y[di])
            decoder_input = output

    # Backward
    loss.backward()

    # Update parameters
    optimizer_encoder.step()
    optimizer_decoder.step()

    return loss.data[0]


# simple predict function
def predict_lstm(model, x_test):
    # x = Variable(x_val, requires_grad=False)
    output = model.forward(x_test)
    return output.data.cpu().numpy()


def predict_enc_dec(encoder, decoder, x_test):
    # x = Variable(x_val, requires_grad=False)
    batch_size = x_test.size()[1]
    encoder_hidden = encoder.initHidden(batch_size)
    encoder_outputs, encoder_hidden = encoder.forward(x_test, encoder_hidden)

    decoder_hidden = encoder_hidden

    output, _ = decoder.forward(encoder_outputs, decoder_hidden)

    return output.data.cpu().numpy()

def predict_enc_attdec(encoder, decoder, x_test):
    # x = Variable(x_val, requires_grad=False)
    batch_size = x_test.size()[1]
    encoder_hidden = encoder.initHidden(batch_size)
    encoder_outputs, encoder_hidden = encoder.forward(x_test, encoder_hidden)

    # prepare the rest of the decoder inputs
    decoder_input = Variable(torch.zeros(1, decoder.hidden_dim)).cuda()
    decoder_context = Variable(torch.zeros(1, decoder.hidden_dim)).cuda()
    decoder_hidden = encoder_hidden

    output, _, _, _ = decoder.forward(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

    return output.data.cpu().numpy()

def as_minutes(s):
    m = np.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))