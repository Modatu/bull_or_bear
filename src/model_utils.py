# utility functions and classes for model.py

# simple LSTMnet class for many to one mapping
class LSTMNet_simple_mto(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet_simple_mto, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        fx, _ = self.lstm.forward(x, (h0, c0))

        return self.linear.forward(fx[-1])

 # simple LSTMnet class for many to many mapping
class LSTMNet_simple_mtm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMNet_simple_mtm, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x):
        batch_size = x.size()[1]
        h0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        c0 = Variable(torch.cuda.FloatTensor(1, batch_size, self.hidden_dim).fill_(0), requires_grad=False)
        fx, _ = self.lstm.forward(x, (h0, c0))

        return self.linear.forward(fx)

# simple training function
def train(model, loss, optimizer, x,y):

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

# simple predict function
def predict(model, x_test):
    # x = Variable(x_val, requires_grad=False)
    output = model.forward(x_test)
    return output.data.cpu().numpy()