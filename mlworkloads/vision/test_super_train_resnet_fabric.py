import sys

import time
from pathlib import Path
from typing import Optional, Tuple, Union
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
from super_dl.cache_coordinator_client import CacheCoordinatorClient
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
    fabric.print(f"Job Id: {hparams.job_id}")
    fabric.launch(main, resume=resume, hparams=hparams)
    #main(fabric,resume=resume, hparams=hparams)


def main(fabric: Fabric, resume: Union[bool, Path],hparams:Namespace) -> None:
   
    # Register job with the cache coordinator service if 'use_super' is True
    if hparams.use_super:
        cache_coordinator_client = CacheCoordinatorClient(server_address=hparams.gprc_server_address)
        cache_coordinator_client.register_job(job_id= hparams.job_id, data_dir=hparams.data_dir, source_system='local')
    
    # Log hyperparameters
    #fabric.loggers[0].log_hyperparams(vars(hparams))
    
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
    train_dataloader = initialize_dataloader(hparams, transformations, 'train', cache_coordinator_client if hparams.use_super else None)

    # Initialize state dictionary
    state = {"model": model, "optimizer": optimizer, "scheduler": scheduler, "hparams": hparams, "iter_num": 0, "step_count": 0}

    # Initialize validation DataLoader if needed and call the run_training function
    if not hparams.train_only:
        val_dataloader = initialize_dataloader(hparams, transformations, 'val', cache_coordinator_client if hparams.use_super else None)
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

def initialize_dataloader(hparams:Namespace, transformations:transforms.Compose, prefix: str, cache_coordinator_client = None) -> DataLoader:

    dataset = SUPERVDataset(
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


    
    train_logger = SuperDLLogger(root_dir=hparams.log_dir,rank=fabric.local_rank,prefix='train',
                                 flush_logs_every_n_steps=hparams.log_interval,
                                 name=hparams.exp_name,
                                 version=hparams.exp_version)
    
    train_logger.log_hyperparams(vars(hparams)) #only rank-0 will do this

    train_epoch_total_time = AverageMeter("batch", ":6.3f")
    train_epoch_dataload_time = AverageMeter("data", ":6.3f")
    train_epoch_compute_time = AverageMeter("compute", ":6.3f")
    train_epoch_losss = AverageMeter("Loss", ":.4e")
    train_epoch_top1_acc = AverageMeter("Acc@1", ":6.2f")
    train_epoch_top5_acc = AverageMeter("Acc@5", ":6.2f")
    train_epoch_num_samples = AverageMeter("Samples", ":6.3f")
    train_epoch_total_batches = AverageMeter("num_batches", ":6.3f")
    
    train_best_prec1 = 0
    train_best_prec5 = 0
    train_best_loss = float('inf')  # Initialize with positive infinity


    if not hparams.train_only:
        
        val_logger = SuperDLLogger(root_dir=hparams.log_dir,rank=fabric.local_rank,prefix='val',
                                 flush_logs_every_n_steps=hparams.log_interval,
                                 name=hparams.exp_name,
                                 version=hparams.exp_version)
        
        val_epoch_total_time = AverageMeter("batch", ":6.3f")
        val_epoch_dataload_time = AverageMeter("data", ":6.3f")
        val_epoch_compute_time = AverageMeter("compute", ":6.3f")
        val_epoch_losss = AverageMeter("Loss", ":.4e")
        val_epoch_top1_acc = AverageMeter("Acc@1", ":6.2f")
        val_epoch_top5_acc = AverageMeter("Acc@5", ":6.2f")
        val_epoch_num_samples = AverageMeter("Samples", ":6.3f")
        val_epoch_total_batches = AverageMeter("num_batches", ":6.3f")
        val_best_prec1 = 0
        val_best_prec5 = 0
        val_best_loss = float('inf')  # Initialize with positive infinity


    for epoch in range(0, hparams.max_epochs):
        try:
            train_epoch_metrics = train(fabric, state, train_dataloader, epoch, train_logger)

            train_epoch_total_time.update(train_epoch_metrics['total_time'])
            train_epoch_dataload_time.update(train_epoch_metrics['dataload_time'])
            train_epoch_compute_time.update(train_epoch_metrics['compute_time'])
            train_epoch_losss.update(train_epoch_metrics['loss'])
            train_epoch_top1_acc.update(train_epoch_metrics['top1'])
            train_epoch_top5_acc.update(train_epoch_metrics['top5'])
            train_epoch_num_samples.update(train_epoch_metrics['total_samples'])
            train_epoch_total_batches.update(train_epoch_metrics['total_batches'])
            state["scheduler"].step()

            train_best_prec1 = max(train_best_prec1, train_epoch_losss.val)
            train_best_prec5 = max(train_best_prec5, train_epoch_top1_acc.val)
            train_best_loss = min(train_best_loss, train_epoch_top5_acc.val)

        except Exception as e:
            fabric.print("Error in training routine!")
            fabric.print(e)
            fabric.print(e.__class__.__name__)
            break
        
        if not hparams.train_only:
            try:
                val_epoch_metrics = validate(fabric, state, val_dataloader, epoch, val_logger)
                val_epoch_total_time.update(val_epoch_metrics['total_time'])
                val_epoch_dataload_time.update(val_epoch_metrics['dataload_time'])
                val_epoch_compute_time.update(val_epoch_metrics['compute_time'])
                val_epoch_losss.update(val_epoch_metrics['loss'])
                val_epoch_top1_acc.update(val_epoch_metrics['top1'])
                val_epoch_top5_acc.update(val_epoch_metrics['top5'])
                val_epoch_num_samples.update(val_epoch_metrics['total_samples'])
                val_epoch_total_batches.update(val_epoch_metrics['total_batches'])

                val_best_prec1 = max(val_best_prec1, val_epoch_losss.val)
                val_best_prec5 = max(val_best_prec5, val_epoch_top1_acc.val)
                val_best_loss = min(val_best_loss, val_epoch_top5_acc.val)

            except Exception as e:
                fabric.print("Error in validation routine!")
                fabric.print(e)
                fabric.print(e.__class__.__name__)
                break

        
        if hparams.always_save_checkpoint:
            fabric.save(state["model"].state_dict(), "resnet_cnn.pt")
    
    # Record job level metrics
    train_logger.log_metrics(metrics={
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        "total_samples": train_epoch_num_samples.sum, 
                        #"batch_idx": batch_idx,
                        "total_batches": train_epoch_total_batches.sum,
                        "total_epochs": train_epoch_total_time.count,
                        "total_time": train_epoch_total_time.sum,
                        "dataloading_time": train_epoch_dataload_time.sum,
                        "compute_time": train_epoch_compute_time.sum,
                        "throughput/samples_per_sec": train_epoch_num_samples.sum / train_epoch_total_time.sum,
                        "throughput/batches_per_sec": train_epoch_total_batches.sum / train_epoch_total_time.sum,
                        "loss": train_best_loss,
                        "top1": train_best_prec1,
                        "top5": train_best_prec5,
                        }, step=None, metric_level='job')

    if not hparams.train_only:
           val_logger.log_metrics(metrics={
                        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        "total_samples": val_epoch_num_samples.sum, 
                        #"batch_idx": batch_idx,
                        "total_batches": val_epoch_total_batches.sum,
                        "total_epochs": val_epoch_total_time.count,
                        "total_time": val_epoch_total_time.sum,
                        "dataloading_time": val_epoch_dataload_time.sum,
                        "compute_time": val_epoch_compute_time.sum,
                        "throughput/samples_per_sec": val_epoch_num_samples.sum / val_epoch_total_time.sum,
                        "throughput/batches_per_sec": val_epoch_total_batches.sum / val_epoch_total_time.sum,
                        "loss": val_best_loss,
                        "top1": val_best_prec1,
                        "top5": val_best_prec5,
                        }, step=None, metric_level='job')




def train(fabric: Fabric, state: dict, train_dataloader: DataLoader,epoch: int, logger:SuperDLLogger) -> None:
    
    model:torch.nn.Module = state["model"]
    optimizer:torch.optim.SGD = state["optimizer"]
    hparams = state["hparams"]

    batch_total_time = AverageMeter("batch", ":6.3f")
    batch_dataload_time = AverageMeter("load", ":6.3f")
    batch_compute_time = AverageMeter("compute", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    total_samples_seen = 0

    # switch to train mode
    #fabric.print("training ...")
    model.train()
    
    total_batches = min(hparams.max_minibatches_per_epoch, len(train_dataloader)) if hparams.max_minibatches_per_epoch else len(train_dataloader)

    end = time.perf_counter()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        #no need to call `.to(device)` on the data, target
        batch_dataload_time.update(time.perf_counter() - end)
        #-----------------Stop data, start compute------#
        if hparams.data_profile:
            torch.cuda.synchronize() # Synchronize before timing GPU operations  
        
        compute_start = time.perf_counter()
        
        if hparams.accelerator == 'gpu':
            # Forward pass
            output = model(input)

            # loss calculation
            loss = F.cross_entropy(output, target)
        
            #Backward pass
            optimizer.zero_grad() # Zero the gradients
            fabric.backward(loss)  # instead of loss.backward()  #Computes gradients
            optimizer.step()  # Update model parameters
        
            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(to_python_float(loss.data), input.size(0))
            top1.update(to_python_float(prec1), input.size(0))
            top5.update(to_python_float(prec5), input.size(0))
        else:
           #simulate training for now
           time.sleep(0.2)
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
                            "epoch": epoch, 
                            #"batch_idx": batch_idx,
                            "total_samples": input.size(0),
                            "total_time": batch_total_time.val,
                            "dataload_time": batch_dataload_time.val,
                            "compute_time": batch_compute_time.val,
                            "loss": losses.val,
                            "top1": top1.val,
                            "top5": top5.val,
                            }
                            ,step=batch_idx, metric_level='batch')

        if hparams.max_minibatches_per_epoch and batch_idx >= hparams.max_minibatches_per_epoch-1:
            #end epoch early based on num_minibacthes that have been processed 
            break

        if batch_idx == len(train_dataloader) - 1:
            break

        end = time.perf_counter()
    
    # Record epoch level metrics
    epoch_metrics = {   "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        "idx": epoch, 
                        #"batch_idx": batch_idx,
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
                        }

    logger.log_metrics(metrics=epoch_metrics, step=None, metric_level='epoch')

    return epoch_metrics




def validate(fabric: Fabric, state: dict, val_dataloader: DataLoader, epoch: int, logger:SuperDLLogger,) -> None:
    
    model:torch.nn.Module = state["model"]
    hparams = state["hparams"]

    batch_total_time = AverageMeter("batch", ":6.3f")
    batch_dataload_time = AverageMeter("load", ":6.3f")
    batch_compute_time = AverageMeter("compute", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    total_samples_seen = 0

    # switch to evaluate mode
    #fabric.print("Validating ...")
    model.eval()
    total_batches = min(hparams.max_minibatches_per_epoch, len(val_dataloader)) if hparams.max_minibatches_per_epoch else len(val_dataloader)

    end = time.perf_counter()

    for batch_idx, (input, target) in enumerate(val_dataloader):
        
        batch_dataload_time.update(time.perf_counter() - end)

        if hparams.data_profile:
            torch.cuda.synchronize() # Synchronize before timing GPU operations  
        # compute output
        compute_start = time.perf_counter()
        output = model(input)
        loss = F.cross_entropy(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(to_python_float(loss.data), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))

        if hparams.data_profile:
            torch.cuda.synchronize()
        
        batch_compute_time.update(time.perf_counter() - compute_start)
        batch_total_time.update(time.perf_counter() - end)
        total_samples_seen += input.size(0)
        
        logger.log_metrics({"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            "epoch": epoch, 
                            #"batch_idx": batch_idx,
                            "total_samples": input.size(0),
                            "total_time": batch_total_time.val,
                            "dataload_time": batch_dataload_time.val,
                            "compute_time": batch_compute_time.val,
                            "loss": losses.val,
                            "top1": top1.val,
                            "top5": top5.val,
                            }
                            ,step=batch_idx, metric_level='batch')

        if hparams.max_minibatches_per_epoch and batch_idx >= hparams.max_minibatches_per_epoch-1:
            #end epoch early based on num_minibacthes that have been processed 
            break

        if batch_idx == len(val_dataloader) - 1:
            break
 
        end = time.perf_counter()

    # Record epoch level metrics
    epoch_metrics = {   "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        "idx": epoch, 
                        #"batch_idx": batch_idx,
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
                        }
    logger.log_metrics(metrics=epoch_metrics, step=None, metric_level='epoch')

    return epoch_metrics

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