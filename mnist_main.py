from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch
from torch import nn
from torch import optim

import syft as sy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
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
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def train_me():
    # A Toy Dataset
    data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
    target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

    # A Toy Model
    model = nn.Linear(2,1)

    # Training Logic
    opt = optim.SGD(params=model.parameters(),lr=0.1)
    for iter in range(20):

        # 1) erase previous gradients (if they exist)
        opt.zero_grad()

        # 2) make a prediction
        pred = model(data)

        # 3) calculate how much we missed
        loss = ((pred - target)**2).sum()

        # 4) figure out which weights caused us to miss
        loss.backward()

        # 5) change those weights
        opt.step()

        # 6) print our progress
        print(loss.data)



    #train()
    hook = sy.TorchHook(torch)



    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")

    # A Toy Dataset
    data = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True)
    target = torch.tensor([[0],[0],[1],[1.]], requires_grad=True)

    # get pointers to training data on each worker by
    # sending some training data to bob and alice
    data_bob = data[0:2]
    target_bob = target[0:2]

    data_alice = data[2:]
    target_alice = target[2:]

    # Iniitalize A Toy Model
    model = nn.Linear(2,1)

    data_bob = data_bob.send(bob)
    data_alice = data_alice.send(alice)
    target_bob = target_bob.send(bob)
    target_alice = target_alice.send(alice)

    # organize pointers into a list
    datasets = [(data_bob,target_bob),(data_alice,target_alice)]

    opt = optim.SGD(params=model.parameters(),lr=0.1)

def data_transforms():
    return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

def build_loaders():
    kwargs = {'num_workers': 1, 'pin_memory': True, 'batch_size': 256 }
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data',
        train=True, download=True, transform=data_transforms()), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data',
        train=False, transform=data_transforms()), shuffle=True, **kwargs)
    return train_loader, test_loader

def mnist():
    device = torch.device("cuda") # if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True, 'batch_size': 256 }
    train_loader = torch.utils.data.DataLoader( datasets.MNIST('../data',
        train=True, download=True, transform=data_transforms()), shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader( datasets.MNIST('../data',
        train=False, transform=data_transforms()), shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    epochs = 1
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


def mnist_fl():
    device = torch.device("cuda") # if use_cuda else "cpu")
    train_loader, test_loader = build_loaders()
    hook = sy.TorchHook(torch)

    bob = sy.VirtualWorker(hook, id="bob")
    alice = sy.VirtualWorker(hook, id="alice")
    print(bob, alice)


if __name__ == '__main__':
    mnist_fl()



