import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.loggers import CSVLogger
#from lightning.fabric.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader, IterableDataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))
from gpt.config import Config
from gpt.model import GPT, Block
from gpt.speed_monitor import SpeedMonitorBase, estimate_flops, measure_flops
from gpt.speed_monitor import SpeedMonitorFabric as SpeedMonitor
from gpt.utils import chunked_cross_entropy, get_default_supported_precision, num_parameters, AverageMeter

model_name = "pythia-70m"
name = "openwebtext"
out_dir = Path("mlworkloads/nlp/out") / name
data_dir = Path("mlworkloads/nlp/data") / name
log_dir = Path("mlworkloads/nlp/logs")

save_interval = 1000
eval_interval = 1000
eval_iters = 1 #org = 100
log_interval = 1

# Hyperparameters
learning_rate = 6e-4
batch_size = 3 #org = 125
micro_batch_size = 3
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5
data_profile = True
hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
logger = CSVLogger(log_dir, name, flush_logs_every_n_steps=log_interval)
always_save_checkpoint = False

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    



def setup(devices: int = 1, precision: Optional[str] = None, resume: Union[bool, Path] = False) -> None:
    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(accelerator="gpu",devices=devices, strategy=strategy, precision=precision, loggers=logger)
    fabric.print(hparams)
    fabric.launch(main, resume=resume)


def main(fabric: L.Fabric, resume: Union[bool, Path]) -> None:
    
    logger.log_hyperparams(hparams)

    speed_monitor = SpeedMonitor(fabric, window_size=50, time_unit="seconds")

    if fabric.global_rank == 0 and always_save_checkpoint:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    config = Config.from_name(model_name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
        model.apply(model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_data, val_data = load_datasets(data_dir, max_seq_length=model.max_seq_length)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=0)
    val_dataloader = DataLoader(val_data, batch_size=micro_batch_size, num_workers=0)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    
    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = sorted(out_dir.glob("*.pth"))[-1]
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader, speed_monitor)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    speed_monitor: SpeedMonitorBase,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    #validate(fabric, model, val_dataloader)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `SpeedMonitor(flops_per_batch=estimated_flops)` instead
        estimated_flops = estimate_flops(meta_model) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.max_seq_length))
        measured_flops = measure_flops(meta_model, x)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x
   
    batch_time = AverageMeter()
    data_time = AverageMeter()
    comp_time = AverageMeter()
    losses = AverageMeter()

    total_lengths = 0
    total_t0 = time.perf_counter()
 
    train_iter = iter(train_dataloader)

    end = time.perf_counter()
    dataset_time = compute_time = 0

    for state["iter_num"] in range(state["iter_num"], max_iters):

        # determine and set the learning rate for this iteration
        lr = get_lr(state["iter_num"]) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids, targets = next(train_iter)
        
        # measure data loading time
        data_time.update(time.perf_counter() - end)
        dataset_time += (time.perf_counter() - end)
        #-----------------Stop data, start compute------#
        if data_profile:
            torch.cuda.synchronize()
        compute_start = time.perf_counter()
        #-----------------------------------------------# 
        # compute output
        is_accumulating = (state["iter_num"] + 1) % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)
        
        losses.update(to_python_float(loss.data), input_ids.size(0))

        if not is_accumulating:
            # compute gradient and do SGD step
           fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
           optimizer.step()
           optimizer.zero_grad()
           state["step_count"] += 1

        torch.cuda.synchronize()
        #-----------------Stop compute------#
        comp_time.update(time.perf_counter() - compute_start)
        compute_time += (comp_time.val)

        batch_time.update(time.perf_counter() - end)

        total_lengths += input_ids.size(1)
        speed_monitor.on_train_batch_end(
            (state["iter_num"] + 1) * micro_batch_size,
            time.perf_counter() - total_t0,
            # this assumes that device FLOPs are the same and that all devices have the same batch size
            fabric.world_size,
            flops_per_batch=measured_flops,
            lengths=total_lengths,
            loss = losses.val,
            dataloading_time=data_time.val,
            forwrd_bkwrd_step= comp_time.val,
            step_time = batch_time.val
        )
        
        if state["iter_num"] % log_interval == 0:
            fabric.print(
                f"iter {state['iter_num']} step {state['step_count']}: loss {loss.item():.4f}, iter time:"
                f" {(batch_time.val) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and state["step_count"] % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader)
            t1 = time.perf_counter() - t0
            speed_monitor.eval_end(t1)
            fabric.print(f"step {state['iter_num']}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if not is_accumulating and state["step_count"] % save_interval == 0 and always_save_checkpoint:
            checkpoint_path = out_dir / f"iter-{state['iter_num']:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)
        
        end = time.perf_counter()


@torch.inference_mode()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    val_iter = iter(val_dataloader)

    losses = torch.zeros(eval_iters, device=fabric.device)
    for k in range(eval_iters):
        input_ids, targets = next(val_iter)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits, targets, chunk_size=0)
    out = losses.mean()

    model.train()
    return out


def load_datasets(data_dir: Path, max_seq_length: int) -> Tuple["Dataset", "Dataset"]:
    train_data = Dataset(data_dir / "train.bin", max_seq_length)
    val_data = Dataset(data_dir / "val.bin", max_seq_length)
    return train_data, val_data


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, max_seq_length: int):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.max_seq_length, (1,)).item()
            x = torch.from_numpy((data[i : i + self.max_seq_length]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.max_seq_length]).astype(np.int64))
            yield x, y


# learning rate decay scheduler (cosine with warmup)
def get_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)