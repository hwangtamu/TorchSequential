import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class SequentialMNIST(nn.Module):
    def __init__(self, batch_size, hidden_size):
        super(SequentialMNIST, self).__init__()
        self.hidden_dim = hidden_size
        self.lstm = nn.LSTM(28, self.hidden_dim)
        self.hidden2label = nn.Linear(self.hidden_dim, 10)
        self.batch_size = batch_size
        self.model = None

    def forward(self, x):
        x = x.permute(1,2,0,3)[0]
        lstm_out, hidden = self.lstm(x)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs

    # def init_hidden(self):
    #     h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
    #     c0 = Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
    #     return (h0, c0)

    def show_pred(self, x, path=None):
        r"""
        :param x: input in shape [time_step, 1, batch_size, input_dim]
        :return:
        """
        if not self.model:
            if path:
                self.model = torch.load(path)
                self.lstm = self.model.lstm
                self.hidden2label = self.model.hidden2label
            else:
                raise AttributeError("Model not loaded.")
        x = x.permute(1,2,0,3)[0]
        self.eval()
        lstm_out, hidden = self.lstm(x)
        # print(lstm_out.size())
        tmp_lab = []
        tmp_val = []
        for o in lstm_out:
            y = self.hidden2label(o)
            tmp_lab += [y.max(1)[1]]
            tmp_val += [F.softmax(y, dim=1).max(1)[0]]
        val = torch.stack(tmp_val, dim=1)
        lab = torch.stack(tmp_lab, dim=1)
        return val, lab






