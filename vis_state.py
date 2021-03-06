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


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # print(data.size())
        # print(torch.reshape(target,(64, -1)))
        # y_onehot = torch.cuda.FloatTensor(output.size(0),output.size(1))
        # y_onehot.zero_()
        # y_onehot.scatter_(1, torch.reshape(target,(target.size(0), -1)), 1)
        # print(output, y_onehot)
        # loss = loss_function(output, y_onehot)

        #loss = loss_function(output, Variable(target))
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main(hidden=256):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader

    model = SequentialMNIST(256, hidden).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

    torch.save(model, './model/new/'+str(hidden)+'_gru.model')


def lesion_test(n, lesion):
    path = './model/new/' + str(n) + '_gru.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    model = SequentialMNIST(64, n, lesion=lesion).to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # model.get_hidden(data, path)
            output = model.show_pred(data, path)
            # for i in range(target.size(0)):
            #     print(target[i].cpu().numpy(), output[1][i].cpu().numpy())
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return correct

def get_state(n):
    path = './model/new/' + str(n) + '_gru.model'
    data = MNIST()
    train_loader = data.trainloader
    test_loader = data.testloader
    device = torch.device("cuda")
    model = SequentialMNIST(64, n).to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model.get_hidden(data, path)
            arr = output[0].cpu().numpy()
            c = np.rollaxis(arr, 1,0)
            for m in c:
                plt.imshow(m.T, cmap='coolwarm')
                plt.colorbar()
                plt.show()

if __name__ == '__main__':

    get_state(256)
