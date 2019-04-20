import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(256, 600),
            nn.ReLU(),
            nn.Linear(600, 100),
            nn.ReLU(),
            nn.Linear(100,2))
        self.decoder = nn.Sequential(
            nn.Linear(2,100),
            nn.ReLU(),
            nn.Linear(100,600),
            nn.ReLU(),
            nn.Linear(600,256))


    def forward(self,x):
        h = self.encoder(x)
        x_ = self.decoder(h)
        return x_

