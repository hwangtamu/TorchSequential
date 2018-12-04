import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class SequentialMNIST(nn.Module):
    def __init__(self, batch_size):
        super(SequentialMNIST, self).__init__()
        self.hidden_dim = 16
        self.lstm = nn.LSTM(28, self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, 10)
        self.batch_size = batch_size
        # self.hidden = self.init_hidden()

    def forward(self, x):
        #print(x.size())
        x = x.permute(1,2,0,3)[0]
        #x = x.view(28, self.batch_size, -1)
        #print(x.size())
        lstm_out, hidden = self.lstm(x)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs

    # def init_hidden(self):
    #     h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
    #     c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
    #     return (h0, c0)
