import torch

from torchvision import datasets, transforms
import torchvision.transforms as transforms


class MNIST:
    def __init__(self):

        # torchvision.datasets.MNIST outputs a set of PIL images
        # We transform them to tensors
        self.transform = transforms.ToTensor()

        # Load and transform data

        self.trainloader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=1000, shuffle=True)

        self.testloader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=64, shuffle=True)


