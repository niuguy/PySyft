from __future__ import print_function
from PIL import Image

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import torch
from torch import nn
from torch import optim
from torchvision.datasets.mnist import MNIST
import pdb

import syft as sy
"""
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
"""


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

def distribute_dataset(data, workers):
    batch_size = int(data.shape[0] / len(workers))
    n_batches = len(workers)
    for batch_i in range(n_batches - 1):
        batch = data[batch_i * batch_size : (batch_i + 1) * batch_size]
        ptr = batch.send(workers[batch_i])
        ptr.child.garbage_collect_data = False

    batch = data[(n_batches - 1) * batch_size :]
    ptr = batch.send(workers[n_batches - 1])
    ptr.child.garbage_collect_data = False


def data_transforms():
    return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

def build_datasets():
    train = datasets.MNIST('../data', train=True, download=True, transform=data_transforms())
    test = datasets.MNIST('../data', train=False, transform=data_transforms())
    return train, test

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
    print('train')
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print(data, target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))


plan = """
-- setup
    SEND_MODEL {}
    LOSS NLL
    OPT CROSSENT
    TRAIN_WITH 100pct SAMPLES

-- run
    CMD_MODEL train
    BEGIN_BATCH(epochs=20)
    CMD_OPT zero_grad
    CMD_MODEL input TO output
    CMD_LOSS output, label TO loss
    CALL backward loss
    CMD_OPT step
    END_BATCH
    PUBLISH_MODEL
    WAIT 2days
"""

class ClientDataset(data.Dataset):
    def __init__(self, tuples, transform=None):
        self.tuples = tuples
        self.transform = transform

    def __getitem__(self, index):
        img, target  = self.tuples[index]
        img = Image.fromarray(img[0].numpy(), mode='L')
        img = self.transform(img)
        return img, int(target)


    def __len__(self):
        return len(self.tuples)


# method local to worker
def train_model(worker):
    print("me", worker.id)
    targets = {}
    data = {}
    # the stuff that organizes pairs of data
    for id, tensor in worker._objects.items():
        t_id = int(id.split('.')[1])
        if 'ys' in tensor.tags:
            targets[t_id] = tensor
        if 'xs' in tensor.tags:
            data[t_id] = tensor
    pairs = [(data[k], targets[k]) for k in targets.keys() ]
    #labels = [ t for t in worker.search('ys') ]
    #samples = [ t for t in worker.search('xs') ]
    #print(str(labels[0].id_at_location), str(samples[0].id_at_location))
    #print(labels[0].clone().get())
    ds = ClientDataset(pairs, transform = data_transforms())
    kwargs = { 'batch_size': 64 }
    train_loader = torch.utils.data.DataLoader(ds, shuffle=True, **kwargs)
    device = torch.device("cuda")
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    epoch = 1
    train(model, device, train_loader, optimizer, epoch)
"""
bob the worker: 
    xs = [
            []
            []
            []
            []
            ]
    ys = [
            1,
            1,
            3,
            2
            ]
"""



def mnist_fl():
    # FEDERATOR LAND
    train, test  = build_datasets()

    hook = sy.TorchHook(torch)
    workers = [sy.VirtualWorker(hook, id="bob", verbose=True),
            sy.VirtualWorker(hook, id="alice", verbose=True),
            sy.VirtualWorker(hook, id="bobby", verbose=True)]


    # 'plans'
    for w in workers:
        # CORE: figure out how to break up into a series of pre-defined commands
        w._message_router['train'] = train_model


    # each worker has some data
    # setting up the example
    buckets = [ [] for _ in workers ]
    for idx, (tensor, lbl) in enumerate(train):
#        if idx > 1000:
#            break
        # TODO: in this example, each worker has 2 tensors (xs, ys)
        worker = workers[idx % len(workers)]
        tensor.id = f"xs.{idx}"
        lbl.id = f"ys.{idx}"
        tensor.tags = ["xs"]
        lbl.tags = ["ys"]
        worker.register_obj(tensor)
        worker.register_obj(lbl)


    # federated learning server
    #workers[0].bubblegum("train")
    workers[0].send_msg('train')


if __name__ == '__main__':
    mnist_fl()


A FederatedLearningParticipant has a worker
A FederatedLearningParticipant defines th communications protocol (default socket)
A FederatedLearningParticipant has a mode and a status
A FederatedLearningParticipant manages a sample store and plan results
FederatedLearningParticipant(worker)


# Hospital A's ugly cron job

    # connect to the data warehouse
    # data = ... 
    flp = FederatedLearningParticipant(protocol="socket",  data=data)

    # load up a thing
    flp.load_data(new_data)
    
    # connect myself to the federated learning party
    flp.set_mode('client', server='1.1.1.1', auth={}) 

    # publish a new model that I get from my thing to other downstream services
    def downstream_me(new_model):
        # the stuff
    flp.on_receiving_new_model = downstream_me

    flp.set_status('ready_to_train')


    flp.set_status('building_round')


FederatedLearningClient:
        status: 'waiting_for_plan', 'receiving', 'ready_to_train', 'training',
        'sending_results'

FederatedLearningServer:
        status: 'sending_plans', 'building_round', 'receiving_results', 'waiting_for_participants'


bob = FLC.worker.objects = (100,256,256), (100,1)
sally = FLC.worker.objects = (300,256,256), (300,1)

mr_manager ('waiting_for_participants'):

bob, sally ('ready_to_train'): 
mr_manager:
    'send sample metadata'
bob, sally: 
    'send worker metadata shapes'
    (100,256,256), (100,1)
    (300,256,256), (300,1)

mr_manager ('building_round'):
    'bob: retry_in 2days'
    'sally: join_round'

bob ('ready_to_train')
sally ('ready_to_train')

mr_manager ('sending_plans'):
    sally: model
    sally: PLAN

mr_manager ('receiving_results'):

sally ('training')
    runs plan...

sally ('sending results')
sally ('waiting')
