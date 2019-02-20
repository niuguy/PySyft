import time
import time
import os
import asyncio
import websockets
from threading import Thread

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



def start_server():
    async def echo(websocket, path):
        async for message in websocket:
            print(f'RCV: {message}')
            time.sleep(1)
            await websocket.send(message)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    asyncio.get_event_loop().run_until_complete(websockets.serve(echo, 'localhost', 8765))
    print("Starting Federator...\n")
    asyncio.get_event_loop().run_forever()



class FederatedLearningServer:
    def __init__(self, id, connection_params, hook):
        self.port = connection_params['port']
        self.host = connection_params['host']
        self.id = id
        self.worker = sy.VirtualWorker(hook, id=id, verbose=True)
        self.current_status = 'waiting_for_clients'
        self.connected_clients = set()

    def load_data(self, obj):
        self.worker.register_obj(obj)

    def unregister(self, websocket):
        self.connected_clients.remove(websocket)

    async def register(self, websocket):
        # Register.
        self.connected_clients.add(websocket)
        try:
            # Implement logic here.
            await asyncio.wait([ws.send("Hello!") for ws in self.connected_clients])
            await asyncio.sleep(1)
        finally:
            self.connected_clients.remove(websocket)
#            self.unregister(websocket)


    async def notify_state():
        if self.connected_clients:
            message = self.current_status
            await asyncio.wait([client.send(message) for client in self.connected_clients])

    async def responder(self, websocket, path):
        # register(websocket) sends user_event() to websocket
        self.connected_clients.add(websocket)
        try:
            await websocket.send(self.current_status)
            async for message in websocket:
                data = json.loads(message)
                if data['action'] == 'SIGN_UP_FOR_ROUND':
                    await self.current_status
                else:
                    logging.error( "unsupported event: {}", data)
        finally:
            pass

    def start(self):
        async def echo(websocket, path):
            async for message in websocket:
                print(f'RCV[{self.id}]: {message}')
                time.sleep(.1)
                await websocket.send(message)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        asyncio.get_event_loop().run_until_complete(websockets.serve(self.responder, self.host, self.port))
        print("Starting Federator...\n")
        asyncio.get_event_loop().run_forever()


    def start_round(self):
        print(self.connected_clients)
        for cl in self.connected_clients:
            cl.send("OH HAI")
        print("foo")

class FederatedLearningClient:
    def __init__(self, id, server_uri, hook, protocol='websocket'):
        self.id = id
        self.server_uri = server_uri
        self.worker = sy.VirtualWorker(hook, id=id, verbose=True)
        self.websocket = None
        self.loop = asyncio.get_event_loop()

    def load_data(self, obj):
        self.worker.register_obj(obj)

    def connect_to_federator(self):
        yield from websockets.connect(self.server_uri)

    async def consumer_handler(self):
        async with websockets.connect(self.server_uri) as websocket:
            result = await websocket.recv()
            print(f'[{self.id}] - signing up. got {result}')



    async def participate_in_round(self):
        async with websockets.connect(self.server_uri) as websocket:
            websocket.send({ 'action': 'SIGN_UP_FOR_ROUND' })
            result = await websocket.recv()
            print(f'[{self.id}] - signing up. got {result}')

def main():
    hook = sy.TorchHook(torch)
    server = FederatedLearningServer("fed1", { 'host': 'localhost', 'port': 8765 }, hook )

    def start():
        server.start()
    thread = Thread(target=start)
    thread.start()



    train, test  = build_datasets()
    num_workers = 3
    xs = [ [] for _ in np.arange(num_workers) ]
    ys = [ [] for _ in np.arange(num_workers) ]
    print("Loading training set...")
    for idx in np.arange(10):
        (tensor, lbl) = train[idx]
        bucket = idx % num_workers
        xs[bucket].append(tensor)
        ys[bucket].append(lbl)



    clients = [FederatedLearningClient(id=f"worker-{idx}", server_uri='ws://localhost:8765', hook=hook) for idx in np.arange(num_workers)]

#    for client in clients:
#        client.connect_to_federator()
#        asyncio.get_event_loop().run_until_complete(client.participate_in_round())
#
#        def fo():
#            loop = asyncio.new_event_loop()
#            asyncio.set_event_loop(loop)
#            asyncio.get_event_loop().run_until_complete(client.consumer_handler())
#        Thread(target=fo).start()

    while True:
        choice = input(">> ")
        try:
            if choice == "s":
                server.start_round()
        except (ValueError, IndexError):
            pass
"""
"""
if __name__ == "__main__":
    main()
