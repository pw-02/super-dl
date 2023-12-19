import torch
from typing import Iterator, Optional, List, TypeVar, Sized, Union, Iterable
from torch.utils.data import Sampler

T = TypeVar('T')

class MyDataset:
    def __init__(self, xs: List[T], ys: List[T]) -> None:
        self.xs = xs
        self.ys = ys

    def __getitem__(self, idx: int) -> int:
        return idx

    def __len__(self) -> int:
        return len(self.xs)

class SuperBaseSampler(Sampler[int]):
    data_source: Sized

    def __init__(self, data_source: Sized, num_samples: Optional[int] = None, shuffle: bool = True, seed: int = 0) -> None:
        self.data_source = data_source
        self._num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)

        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_seed(self, seed: int) -> None:
        self.seed = seed

class SuperBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool) -> None:
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    batch_id = abs(hash(tuple(batch)))
                    yield batch, batch_id
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    batch_id = abs(hash(tuple(batch)))
                    yield batch, batch_id
                    idx_in_batch = 0
                    batch = [0] * self.batch_size

            if idx_in_batch > 0:
                batch_id = abs(hash(tuple(batch[:idx_in_batch])))
                yield batch[:idx_in_batch], batch_id

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def set_seed(self, seed: int) -> None:
        if isinstance(self.sampler, SuperBaseSampler):
            self.sampler.set_seed(seed)
        else:
            raise ValueError("The underlying sampler must be an instance of SuperBaseSampler")

class SuperSampler(SuperBatchSampler):
    def __init__(self, sampler, batch_size, drop_last, grpc_server_address):
        super(SuperSampler, self).__init__(sampler, batch_size, drop_last)
        self.grpc_server_address = grpc_server_address
        self.grpc_client = None
        self.prefetch_batches = 20

    def __iter__(self) -> Iterator[List[int]]:
        batch_buffer = []
        batch_id_buffer = []
        first_iteration = True
        batch_iter = super().__iter__()

        while True:
            if first_iteration:
                while len(batch_buffer) < self.prefetch_batches * 2:
                    try:
                        batch_indices, batch_id = next(batch_iter)
                        batch_buffer.append(batch_indices)
                        batch_id_buffer.append(batch_id)
                    except StopIteration:
                        break
                print(batch_buffer)
            
            elif len(batch_buffer) <= self.prefetch_batches:
                prefecth_buffer =[]
                prefecth_batch_id_buffer =[]
                for _ in range(self.prefetch_batches):
                    try:
                        batch_indices, batch_id = next(batch_iter)
                        prefecth_buffer.append(batch_indices)
                        prefecth_batch_id_buffer.append(batch_id)
                    except StopIteration:
                        break
                batch_buffer += prefecth_buffer
                batch_id_buffer += prefecth_batch_id_buffer
                print(prefecth_buffer)
                print(batch_buffer)

            if not batch_buffer:
                break

            batch = batch_buffer.pop(0)
            batch_id = batch_id_buffer.pop(0)

            yield batch, batch_id
            first_iteration = False   

    def set_grpc_server_address(self, grpc_server_address):
        self.grpc_server_address = grpc_server_address
        # Update the gRPC client with the new server address
        if self.grpc_client:
            self.grpc_client.set_server_address(grpc_server_address)

# Usage example
if __name__ == "__main__":
    xs = list(range(100))
    ys = list(range(100, 1000))
    dataset = MyDataset(xs, ys)
    base_sampler = SuperBaseSampler(dataset, shuffle=False)

    super_grpc_batch_sampler = SuperSampler(base_sampler, 1, False, None)

    train_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=None, sampler=super_grpc_batch_sampler)

    for epoch in range(2):
        train_loader.sampler.set_seed(epoch)
        print(f'Epoch: {epoch}:')
        for batch_indices, batch_id in train_loader:
            print(f'Batch ID: {batch_id}, Batch Indices: {batch_indices}')
