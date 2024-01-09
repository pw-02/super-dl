import sys

import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from pytorch.loggers.super_logger import SuperDLLogger
from misc.args import parse_args
from misc.utils import get_default_supported_precision, num_parameters, AverageMeter, ProgressMeter
from pytorch.datasets.super_dataset import SUPERVDataset
from pytorch.samplers.super_sampler import SUPERSampler
import torch.nn.functional as F
import os
from torchvision import models, transforms
from argparse import Namespace
from typing import List
# Import CacheCoordinatorClient
from SuperDL.cache_coordinator_client import CacheCoordinatorClient
from lightning.fabric import Fabric, seed_everything
from datetime import datetime


import logging
log = logging.getLogger(__name__)

def to_python_float(t:torch.Tensor)-> float:
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
  
def setup(config_file:str, devices: int, precision: Optional[str], resume: Union[bool, Path]) -> None:
    hparams = parse_args(config_file=config_file)
    hparams.job_id = os.getpid()
    hparams.exp_version = get_next_exp_version(root_dir=hparams.log_dir, name=hparams.exp_name)
    precision = precision or get_default_supported_precision(training=True)
    
    if devices > 1:
        strategy = FSDPStrategy(state_dict_type="full",limit_all_gathers=True,cpu_offload=False,)
    else:
        strategy = "auto"

    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `lightning run model --help`
    fabric = Fabric(accelerator=hparams.accelerator,devices=hparams.devices, strategy=strategy, precision=precision)
    #fabric.print(hparams)
    
    end = time.perf_counter()

    fabric.print(f"Job Id: {hparams.job_id}")
    fabric.launch(main, resume=resume, hparams=hparams)

    total_duration = time.perf_counter() - end

    fabric.print(f"Time for script to finish: {total_duration}")

    #main(fabric,resume=resume, hparams=hparams)


def main(fabric: Fabric, resume: Union[bool, Path],hparams:Namespace) -> None:
   
    # Register job with the cache coordinator service if 'use_super' is True
    if hparams.use_super:
        cache_coordinator_client = CacheCoordinatorClient(server_address=hparams.gprc_server_address)
        cache_coordinator_client.register_job(job_id= hparams.job_id, data_dir=hparams.data_dir, source_system='local')
    
    # Create save directory if needed
    if fabric.global_rank == 0 and hparams.always_save_checkpoint:
        hparams.save_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    fabric.seed_everything(1337, workers=True)
    
    # Load model
    t0 = time.perf_counter()
    model = initialize_model(fabric, hparams.arch)
    
    # Print model information
    fabric.print(f"Time to instantiate {hparams.arch} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {hparams.arch} model: {num_parameters(model):,}")

    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))
    
    # call `setup` to prepare for model / optimizer for distributed training. The model is moved automatically to the right device.
    model, optimizer = fabric.setup(model,optimizer)  

    #Initialize data transformations
    transformations = initialize_transformations()

    # Initialize train DataLoader
    train_dataloader = initialize_dataloader(fabric, hparams, transformations, 'train', cache_coordinator_client if hparams.use_super else None)

    # Initialize state dictionary
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "hparams": hparams, "iter_num": 0, "step_count": 0}

    # Initialize validation DataLoader if needed and call the run_training function
    if not hparams.train_only:
        val_dataloader = initialize_dataloader(fabric, hparams, transformations, 'val', cache_coordinator_client if hparams.use_super else None)
        train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    else:
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

    run_training(fabric, state, train_dataloader, val_dataloader if not hparams.train_only else None, hparams)

    #create_job_report(job_id =hparams.job_id,log_out_folder=fabric.loggers[0].log_dir)


def initialize_model(fabric: Fabric, arch: str) -> torch.nn.Module: 
    with fabric.init_module(empty_init=True): #model is instantiated with randomly initialized weights by default.
        model: torch.nn.Module = models.get_model(arch)
    return model

def initialize_transformations() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.ToTensor(), normalize])
    return transformations

def initialize_dataloader(fabric: Fabric, hparams:Namespace, transformations:transforms.Compose, prefix: str, cache_coordinator_client = None) -> DataLoader:

    dataset = SUPERVDataset(
        fabric=fabric,
        source_system=hparams.source_system,
        cache_host=hparams.cache_host if hparams.use_super else None,
        data_dir=hparams.data_dir,
        prefix=prefix,
        transform=transformations
    )
    sampler = SUPERSampler(
        data_source=dataset,
        job_id=hparams.job_id,
        grpc_client=cache_coordinator_client,
        shuffle=True,
        seed=0,
        batch_size=hparams.batch_size,
        drop_last=False,
        prefetch_look_ahead=30
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=hparams.num_workers)
    return dataloader


def run_training(fabric: Fabric, state: dict, train_dataloader: DataLoader, val_dataloader: DataLoader, hparams:Namespace) -> None:

    def initialize_logger(prefix: str) -> SuperDLLogger:
        return SuperDLLogger(
            root_dir=hparams.log_dir,
            rank=fabric.local_rank,
            prefix=prefix,
            flush_logs_every_n_steps=hparams.log_interval,
            name=hparams.exp_name,
            version=hparams.exp_version
        )
    
    def record_job_level_metrics(logger: SuperDLLogger, total_time: AverageMeter, dataload_time: AverageMeter,
                       compute_time: AverageMeter, loss: AverageMeter, top1: AverageMeter, top5: AverageMeter,
                       total_samples: AverageMeter, total_batches: AverageMeter, metric_level: str) -> None:
         logger.log_metrics(metrics={
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            "total_samples": total_samples.sum,
            "total_batches": total_batches.sum,
            "total_epochs": total_time.count,
            "total_time": total_time.sum,
            "dataloading_time": dataload_time.sum,
            "compute_time": compute_time.sum,
            "throughput/samples_per_sec": total_samples.sum / total_time.sum,
            "throughput/batches_per_sec": total_batches.sum / total_time.sum,
            "loss": loss.min,
            "top1": top1.max,
            "top5": top5.max,
            "rank": fabric.local_rank

            }, step=None, metric_level=metric_level)
        
    train_logger = initialize_logger('train')
    train_logger.log_hyperparams(vars(hparams))  # Only rank-0 will do this

    train_metrics = {
        'total_time': AverageMeter("batch", ":6.3f"),
        'dataload_time': AverageMeter("data", ":6.3f"),
        'compute_time': AverageMeter("compute", ":6.3f"),
        'loss': AverageMeter("Loss", ":.4e"),
        'top1': AverageMeter("Acc@1", ":6.2f"),
        'top5': AverageMeter("Acc@5", ":6.2f"),
        'total_samples': AverageMeter("Samples", ":6.3f"),
        'total_batches': AverageMeter("num_batches", ":6.3f"),
        }

    
    if not hparams.train_only:
            val_logger = initialize_logger('val')
            val_metrics = {
            'total_time': AverageMeter("batch", ":6.3f"),
            'dataload_time': AverageMeter("data", ":6.3f"),
            'compute_time': AverageMeter("compute", ":6.3f"),
            'loss': AverageMeter("Loss", ":.4e"),
            'top1': AverageMeter("Acc@1", ":6.2f"),
            'top5': AverageMeter("Acc@5", ":6.2f"),
            'total_samples': AverageMeter("Samples", ":6.3f"),
            'total_batches': AverageMeter("num_batches", ":6.3f"),
        }
        
    for epoch in range(hparams.max_epochs):
            
            train_epoch_metrics = train(fabric, state, train_dataloader, epoch, train_logger)

            for key in train_metrics:
                train_metrics[key].update(train_epoch_metrics[key])
            
            if hparams.accelerator == 'gpu':
                state["scheduler"].step()
            
            if not hparams.train_only:
                val_epoch_metrics = validate(fabric, state, val_dataloader, epoch, val_logger)
                for key in val_metrics:
                    val_metrics[key].update(val_epoch_metrics[key])
            
            if hparams.always_save_checkpoint:
                fabric.save(state["model"].state_dict(), "resnet_cnn.pt")
    
    # Record job level metrics
    record_job_level_metrics(train_logger, **train_metrics, metric_level='job')
    if not hparams.train_only:
        record_job_level_metrics(val_logger, **val_metrics, metric_level='job')      


def process_data(fabric: Fabric, state: dict, dataloader: DataLoader, model:torch.nn.Module,
                  optimizer:torch.optim.SGD, logger:SuperDLLogger, epoch, is_training=True):
    
    model.train(is_training)
    hparams = state["hparams"]

    batch_total_time = AverageMeter("Total Time", ":6.3f")
    batch_dataload_time = AverageMeter("DataLoad Time", ":6.3f")
    batch_compute_time = AverageMeter("Compute Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    total_samples_seen = 0
    
    total_batches = min(hparams.max_minibatches_per_epoch, len(dataloader)) if hparams.max_minibatches_per_epoch else len(dataloader)
    progress = ProgressMeter(
        total_batches, [batch_total_time, batch_dataload_time, batch_compute_time, losses, top1, top5], prefix="Rank[{}]\tEpoch: [{}]".format(fabric.local_rank,epoch)
    )
    end = time.perf_counter()
    
    for batch_idx, (input, target, batch_id) in enumerate(dataloader):
        batch_dataload_time.update(time.perf_counter() - end)
        
        if hparams.data_profile:
            torch.cuda.synchronize()

        compute_start = time.perf_counter()
        if hparams.accelerator == 'gpu':

            # Forward pass
            output = model(input)

            # loss calculation
            loss = F.cross_entropy(output, target)
            
            if is_training:
                optimizer.zero_grad() # zero the gradients
                fabric.backward(loss)  # instead of loss.backward()  #Computes gradients
                optimizer.step()  # Update model parameters
                
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(to_python_float(loss.data), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))
        else:
            # simulate training for now
            time.sleep(0.1)
            losses.update(0.0)
            top1.update(0.0)
            top5.update(0.0)
    
        if hparams.data_profile:
            torch.cuda.synchronize()
        
        batch_compute_time.update(time.perf_counter() - compute_start)
        # measure elapsed time
        batch_total_time.update(time.perf_counter() - end)
        
        total_samples_seen += input.size(0)

        logger.log_metrics({"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            "id": str(batch_id),
                            "epoch": epoch, 
                            "total_samples": input.size(0),
                            "total_time": batch_total_time.val,
                            "dataload_time": batch_dataload_time.val,
                            "compute_time": batch_compute_time.val,
                            "loss": losses.val,
                            "top1": top1.val,
                            "top5": top5.val,#
                            "rank": fabric.local_rank
                            }
                            , step=batch_idx, metric_level='batch')
        
        if (batch_idx + 1) % hparams.log_interval == 0:
            progress.display(batch_idx)


        if hparams.max_minibatches_per_epoch and batch_idx >= hparams.max_minibatches_per_epoch - 1:
            # end epoch early based on num_minibatches that have been processed 
            break

        if batch_idx == len(dataloader) - 1:
            break

        end = time.perf_counter()    
    
     # Record epoch level metrics
    epoch_metrics = {"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                     "idx": epoch, 
                     "total_batches": batch_total_time.count,
                     "total_samples": total_samples_seen,
                     "total_time": batch_total_time.sum,
                     "dataload_time": batch_dataload_time.sum,
                     "compute_time": batch_compute_time.sum,
                     "throughput/samples_per_sec": total_samples_seen / batch_total_time.sum,
                     "throughput/batches_per_sec": batch_total_time.count / batch_total_time.sum,
                     "loss": losses.avg,
                     "top1": top1.avg,
                     "top5": top5.avg,
                     "rank": fabric.local_rank
                     }

    logger.log_metrics(metrics=epoch_metrics, step=None, metric_level='epoch')

    return epoch_metrics   


def train(fabric: Fabric, state: dict, train_dataloader: DataLoader,epoch: int, logger:SuperDLLogger)-> dict[str, Any]:
    model = state["model"]
    optimizer = state["optimizer"]
    train_epoch_metrics = process_data(fabric, state, train_dataloader, model, optimizer, logger, epoch, is_training=True)
    return train_epoch_metrics

def validate(fabric: Fabric, state: dict, val_dataloader: DataLoader, epoch: int, logger:SuperDLLogger)-> dict[str, Any]:
    model = state["model"]
    val_epoch_metrics = process_data(fabric, state, val_dataloader, model, None, logger, epoch, is_training=False)
    return val_epoch_metrics


def accuracy(output: torch.Tensor, target:torch.Tensor, topk=(1,))-> List[torch.Tensor]:
    """Computes the accuracy over the k top predictions for the specified
    values of k."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def get_next_exp_version(root_dir,name):
    from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

    versions_root = os.path.join(root_dir, name)
    fs = get_filesystem(root_dir)
    if not _is_dir(fs, versions_root, strict=True):
            log.warning("Missing logger folder: %s", versions_root)
            return 0
    
    existing_versions = []
    for d in fs.listdir(versions_root):
        full_path = d["name"]
        name = os.path.basename(full_path)
        if _is_dir(fs, full_path) and name.startswith("version_"):
            dir_ver = name.split("_")[1]
            if dir_ver.isdigit():
                existing_versions.append(int(dir_ver))
    if len(existing_versions) == 0:
        return 0
    return max(existing_versions) + 1


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")

    defaults = {
            "config_file": 'mlworkloads/cfgs/resnet18.yaml',
            'devices': 1,
            'precision': None,
            'resume': False,
        }

    from jsonargparse import CLI
    CLI(setup, set_defaults=defaults)