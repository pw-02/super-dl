import grpc
import torch
import random
import time
import numpy as np
from torch.utils.data import Sampler

import torch
import random
import numpy as np
from torch.utils.data import Sampler

import random
import numpy as np
from torch.utils.data import Sampler

class SimpleBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, seed=42):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.seed = seed

        self.order_of_batches = list(range(dataset_size))
        self.current_position = 0

        # Set the seed for deterministic shuffling
        random.seed(seed)
        np.random.seed(seed)

        # Shuffle the order of batches
        random.shuffle(self.order_of_batches)

    def __iter__(self):
        while self.current_position < self.dataset_size:
            yield self._get_next_batch_indices()

        # Reset position for the next epoch
        self.current_position = 0

    def __len__(self):
        return self.dataset_size // self.batch_size

    def _get_next_batch_indices(self):
        start_position = self.current_position
        end_position = min(start_position + self.batch_size, self.dataset_size)
        batch_indices = self.order_of_batches[start_position:end_position]
        
        # Generate a batch ID based on the indices within the batch
        batch_id = hash(tuple(batch_indices))
        
        # Update position
        self.current_position = end_position

        return batch_indices, batch_id




class MyDataset:
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
    
    def __getitem__(self,idx):
        return idx[0]
    
    def __len__(self):
        return len(self.xs)   


if __name__ == "__main__": 
    
    xs = list(range(1000))
    ys = list(range(100,1000))
    dataset = MyDataset(xs, ys)
    seed = int(torch.empty((), dtype=torch.int64).random_().item())
    seed = 0
    train_sampler = SimpleBatchSampler(len(dataset),batch_size=256, seed=42)
    train_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None, sampler=train_sampler)

    for epoch in range(4):
        print(f'Epoch: {epoch}:')
        for batch_idx, batch_data in enumerate(train_loader):
            print(len(batch_data))


class SUPERBatchSampler(Sampler):

    def __init__(self, dataset_size, batch_size, grpc_server_address='localhost:50051', send_interval=60, batch_count_threshold=500, seed=42):
        from super import cache_management_pb2, cache_management_pb2_grpc
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.grpc_server_address = grpc_server_address
        self.send_interval = send_interval
        self.batch_count_threshold = batch_count_threshold
        self.seed = seed

        self.order_of_batches = list(range(dataset_size))
        self.current_position = 0
        self.batch_count = 0
        self.last_send_time = time.time()

        # Set the seed for deterministic shuffling
        random.seed(seed)
        np.random.seed(seed)

        # Shuffle the order of batches
        random.shuffle(self.order_of_batches)

        # Create the gRPC channel and stub
        self.channel = grpc.insecure_channel(self.grpc_server_address)
        self.stub = cache_management_pb2_grpc.CacheManagementStub(self.channel)

    def __del__(self):
        # Close the gRPC channel when the sampler is deleted
        if hasattr(self, 'channel'):
            self.channel.close()

    def __iter__(self):
        while self.current_position < self.dataset_size:
            yield self._get_next_batch_indices()

        # Optionally, yield remaining batches
        while self.current_position < len(self.order_of_batches):
            yield self._get_next_batch_indices()

    def __len__(self):
        return self.dataset_size // self.batch_size

    def _get_next_batch_indices(self):
        start_position = self.current_position
        end_position = min(start_position + self.batch_size, self.dataset_size)
        batch_indices = self.order_of_batches[start_position:end_position]
        # Generate a batch ID based on the indices within the batch
        batch_id = hash(tuple(batch_indices))
        
        # Update position and reset batch count
        self.current_position = end_position
        self.batch_count = 0
        self.last_send_time = time.time()

        # Update batch count based on the number of batches sent
        self.batch_count += len(batch_indices)

        # Send batches when the threshold or interval is reached
        if self.batch_count >= self.batch_count_threshold or time.time() - self.last_send_time >= self.send_interval:
            self._send_batches_to_grpc(batch_indices)
        
        return batch_indices, batch_id

    def _send_batches_to_grpc(self, batches):
        from super import cache_management_pb2, cache_management_pb2_grpc

        # Prepare the gRPC request
        request = cache_management_pb2.BatchIds(ids=batches)

        # Send the gRPC request
        response = self.stub.PreloadBatches(request)

        # Print the gRPC response (replace with your handling logic)
        print("gRPC PreloadBatches Response:", response.message)
