from mlworkloads.classification.image_classification.samplers import SUPERSampler
from superdl.syncgrpc.client import SuperClient
from typing import Iterator, Optional, List, TypeVar, Sized, Union, Iterable, Dict, Tuple
import random
import functools
from torch.utils.data import DataLoader


T = TypeVar('T')

class SimpleDataset:
    def __init__(self, dataset_id, datset_size=25, num_labels = 10) -> None:
        
        self.dataset_id = dataset_id
        self.samples: Dict[str, List[str]] = {}

        for sample in range(datset_size):
            random_label =  random.randint(1, num_labels)
            self.samples.setdefault(random_label,[]).append(sample)
    
        
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self.samples)
            for blob in self.samples[blob_class]
        ]
    
    def get_samples_for_batch(self, batch_sample_indices):
        samples = []
        for i in  batch_sample_indices:
                samples.append(self._classed_items[i])
        return samples
    

    def __getitem__(self, next_batch):
        batch_indices, batch_id = next_batch
        batch_samples = self.get_samples_for_batch(batch_indices)
        return  batch_id, batch_samples,

    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self.samples.values())


def test_commuication_with_super_server(server_address='localhost:50051', dataset_size=10):
    job_id = 1
    num_epochs = 2
    super_client = SuperClient(server_address)
    dataset = SimpleDataset(dataset_id= 'simple_dataset', datset_size=100, num_labels=2)
    super_client.register_dataset(dataset.dataset_id, None, None, dataset.samples)
    super_client.register_new_job(job_id=job_id, job_dataset_ids=[dataset.dataset_id])

    new_sampler = SUPERSampler(dataset=dataset, job_id=job_id, super_client=super_client, shuffle=False,
                               batch_size=16, drop_last=False, prefetch_lookahead=2)
    
    train_loader = DataLoader(dataset, num_workers=0, batch_size=None, sampler=new_sampler)
    
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}:')
        for batch_id, batch_indices in train_loader:
            print(f'Batch ID: {batch_id}, Batch Indices: {batch_indices}')

if __name__ == '__main__':
    test_commuication_with_super_server()
