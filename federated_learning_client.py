import time
import asyncio
import websockets
import syft as sy

class FederatedLearningClient:
    def __init__(self, id, connection_params, hook, data=(), loop=None):
        self.port = connection_params['port']
        self.host = connection_params['host']
        self.id = id
        self.worker = sy.VirtualWorker(hook, id=id, verbose=True)
        self.current_status = 'ready'
        self.connections = set()
        self.msg_queue = asyncio.Queue()
        self.loop = asyncio.new_event_loop() if loop is None else loop
        self.uri = f'ws://{self.host}:{self.port}'
        for item in data:
            self.worker.register_obj(item)

    def msg(self, msg):
        return f'[{self.id}] {msg}'

    def worker_metadata(self):
        return [ obj.shape for key, obj in self.worker._objects.items() ]

    async def consumer_handler(self, websocket):
        while True:
            if not websocket.open:
                websocket = await websockets.connect(self.uri)
            msg = await websocket.recv()
            print(f'[{self.id} | RCV] {msg}')
            await self.msg_queue.put(msg)


    async def producer_handler(self, websocket):
        while True:
            msg = await self.msg_queue.get()
            if msg == 'STAT':
                await websocket.send(self.msg(self.current_status))
            if msg == 'META':
                await websocket.send(self.msg(self.worker_metadata()))


    async def handler(self, websocket):
        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self.consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self.producer_handler(websocket))

        done, pending = await asyncio.wait([consumer_task, producer_task]
                                        , return_when=asyncio.FIRST_COMPLETED)
        print("Connection closed, canceling pending tasks")
        for task in pending:
            task.cancel()

    async def connect(self):
        async with websockets.connect(self.uri) as websocket:
            while True:
                if not websocket.open:
                    websocket = await websockets.connect(self.uri)
                await self.handler(websocket)

    def start(self):
        asyncio.get_event_loop().run_until_complete(self.connect())
        print("Starting client..\n")


