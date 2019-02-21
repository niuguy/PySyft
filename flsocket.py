import time
import os
import asyncio
import websockets
from multiprocessing import Process
from collections import ChainMap

import numpy as np
import torch
import torch.utils.data as data
from torchvision import datasets, transforms
from torchvision.datasets.mnist import MNIST
import syft as sy

from federated_learning_server import FederatedLearningServer
from federated_learning_client import FederatedLearningClient

def data_transforms():
    return transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])

def build_datasets():
    train = datasets.MNIST('../data', train=True, download=True, transform=data_transforms())
    test = datasets.MNIST('../data', train=False, transform=data_transforms())
    return train, test


def start_proc(participant, kwargs):
    """ helper function for spinning up a websocket participant """
    def target():
        server = participant(**kwargs)
        server.start()
    p = Process(target=target)
    p.start()

async def repl(uri='ws://localhost:8765'):
    async with websockets.connect(uri) as websocket:
        while True:
            if not websocket.open:
                websocket = await websockets.connect(uri)
            cmd = input("\ncmd:  ")
            await websocket.send(cmd)
            resp = await websocket.recv()
            print("<REPL> {}".format(resp))

def main():
    hook = sy.TorchHook(torch)
    kwargs = { "id": "fed1", "connection_params": { 'host': 'localhost', 'port': 8765 }, "hook": hook }
    start_proc(FederatedLearningServer, kwargs)
    time.sleep(1)

    train, test  = build_datasets()
    num_workers = 3
    data_sent = 10
    xs = [ [] for _ in np.arange(num_workers) ]
    ys = [ [] for _ in np.arange(num_workers) ]
    # for now, we save ourselves the trouble of loading ALL the MNIST data:
    for idx in np.arange(data_sent):
        (tensor, lbl) = train[idx]
        bucket = idx % num_workers
        xs[bucket].append(tensor)
        ys[bucket].append(lbl)

    for idx in np.arange(num_workers):
        x = torch.tensor(np.array(xs[idx]))
        x.id = 'xs'
        y = torch.tensor(np.array(ys[idx]))
        y.id = 'ys'
        worker_args = ChainMap({ 'id': f'w{idx}', 'data': (x, y) }, kwargs)
        start_proc(FederatedLearningClient, worker_args)

    # repl for issuing commands
    asyncio.get_event_loop().run_until_complete(repl())

if __name__ == "__main__":
    main()
