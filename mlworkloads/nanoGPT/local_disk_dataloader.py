import os
import random
import tarfile
from functools import partial
from typing import Optional, List, Callable
from itertools import islice
import torch
import tiktoken
import gdown

class OpenWebTextCorpus(torch.utils.data.IterableDataset):

    def __init__(self, tar_filename: str):
        super().__init__()
        self.tar_filename = tar_filename
        if os.path.exists(self.tar_filename) is False:
            OpenWebTextCorpus.download_file()

    @property
    def document_count(self):
        return 8013769
    

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if not worker_info:
            return self.get_stream()
        else:
            return islice(
                self.get_stream(), worker_info.id, None, worker_info.num_workers
            )


def collate_fn(
    batch: List[str],
    encoder: Callable,
    block_size: int,
    dtype: torch.dtype,
    use_dynamic_batching: bool,
):
    encoded = encoder.encode_ordinary_batch(batch)
    max_length = (
        min(block_size, max([len(input_ids) for input_ids in encoded]))
        if use_dynamic_batching
        else block_size
    )
    input_ids = torch.empty((len(batch), max_length), dtype=dtype).fill_(
        encoder.eot_token
    )
    targets = torch.empty((len(batch), max_length), dtype=dtype).fill_(
        encoder.eot_token
    )
    for index in range(len(encoded)):
        block = encoded[index]
        if len(block) - max_length > 0:
            # sample tokens
            start = random.randrange(0, len(block) - max_length)
            block = block[start : start + max_length + 1]
        l = len(block[:-1][:max_length])
        input_ids[index, :l] = torch.tensor(block[:-1][:max_length])
        targets[index, :l] = torch.tensor(block[1:][:max_length])
    return {
        "input_ids": input_ids,
        "targets": targets,
    }


def get_data_loader(
    tar_filename: str,
    block_size: Optional[int] = 1024,
    batch_size: Optional[int] = 4,
    num_workers: Optional[int] = 4,
    prefetch_factor: Optional[int] = 8,
    dtype: Optional[torch.dtype] = torch.int64,
    use_dynamic_batching: Optional[bool] = False,
):
    iterable_dataset = OpenWebTextCorpus(tar_filename=tar_filename)
    encoder = tiktoken.get_encoding("gpt2")
    assert (
        torch.tensor(encoder.max_token_value).type(dtype).item()
        == encoder.max_token_value
    ), "`dtype` does not cover the full range of values for the vocabulary"
    collate_fn_partial = partial(
        collate_fn,
        encoder=encoder,
        block_size=block_size,
        dtype=dtype,
        use_dynamic_batching=use_dynamic_batching,
    )
    return torch.utils.data.DataLoader(
        dataset=iterable_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=collate_fn_partial,
        pin_memory=True,
    )