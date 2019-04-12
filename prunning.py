from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from src.LSTM import SequentialMNIST
from src.data import MNIST
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

#loss_function = nn.MSELoss()
loss_function = nn.CrossEntropyLoss()
bins = 100
hist_range = [-1,1]
labels = [str(x) for x in range(10)]


def get_output(n):
    path = './model/new/'+str(n)+'_lstm.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    x_ = []
    y_ = []
    with torch.no_grad():

        model = SequentialMNIST(64, n).to(device)
        model.load(path)
        w = model.hidden2label.weight.data
        print(list(model.parameters())[-2])
        w = w.cpu().numpy()
        w_2 = np.mean(np.absolute(w), axis=0)
        print(np.argmin(w_2), np.min(w_2))
        print(np.argmax(w_2), np.max(w_2))
        w_sorted = np.argsort(w_2)
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model.get_hidden(data, path)
            print(output[0].size())
            x = output[0][-1].cpu().numpy().T
            y = target.cpu().numpy()
            x_ += list(x[w_sorted[-5]])
            y_ += list(y)
            # print(x.shape, y.shape)
            # plt.scatter(y,x[np.argmax(w_2)], c='b', alpha=0.1)
        #     corr = np.corrcoef([*x,y])[-1][:-1]
        #     plt.scatter(list(range(256)),corr,c='b',alpha=0.1)
        x_ = np.array(x_)
        y_ = np.array(y_)
        datasets = [x_[y_==t] for t in range(10)]
        binned_data_sets = [
            np.histogram(d, range=hist_range, bins=100)[0]
            for d in datasets
            ]
        binned_maximums = np.max(binned_data_sets, axis=1)
        print(binned_maximums)
        x_locations = np.arange(0, sum(binned_maximums), sum(binned_maximums)//10)
        bin_edges = np.linspace(hist_range[0], hist_range[1], 100 + 1)
        centers = 0.5 * (bin_edges + np.roll(bin_edges, 1))[:-1]
        heights = np.diff(bin_edges)

        # Cycle through and plot each histogram
        fig, ax = plt.subplots()
        for x_loc, binned_data in zip(x_locations, binned_data_sets):
            lefts = x_loc - 0.5 * binned_data
            ax.barh(centers, binned_data, height=heights, left=lefts, color='b')

        ax.set_xticks(x_locations)
        ax.set_xticklabels(labels)

        ax.set_ylabel("Activation")
        ax.set_xlabel("Label")

        plt.grid()
        plt.show()




def lesion_test(n, lesion):
    path = './model/new/' + str(n) + '_lstm.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    model = SequentialMNIST(64, n).to(device)
    model.load(path)
    w = model.hidden2label.weight.data
    w = w.cpu().numpy()
    w_2 = np.mean(np.absolute(w), axis=0)
    w_sorted = np.argsort(w_2)
    blocked = w_sorted[:256-lesion]
    model2 = SequentialMNIST(64, n, blocked=blocked).to(device)
    model2.load(path, blocked)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # model.get_hidden(data, path)
            # output = model.show_pred(data, path)
            # for i in range(target.size(0)):
            #     print(target[i].cpu().numpy(), output[1][i].cpu().numpy())
            output = model2(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct

if __name__ == '__main__':
    get_output(256)
    # for i in [4,6,8,10,12,16,32,64,128,256]:
    #     main(i)
    # lesion_test(256, 4)