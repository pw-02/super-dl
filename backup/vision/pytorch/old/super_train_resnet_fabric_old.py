import sys

import time
from pathlib import Path
from typing import Optional, Tuple, Union
import lightning as L
import torch
from lightning.pytorch.loggers import CSVLogger
from lightning.fabric.strategies import FSDPStrategy
from torch.utils.data import DataLoader
from misc.args import parse_args
from misc.utils import get_default_supported_precision, num_parameters, AverageMeter, ProgressMeter
from misc.ml_training_logger import MLTrainingLogger, create_job_report
from pytorch.datasets.super_dataset import SUPERVDataset
from pytorch.samplers.super_sampler import SUPERSampler
import torch.nn.functional as F
import os
from torchvision import models, transforms
from argparse import Namespace
from typing import List
# Import CacheCoordinatorClient
from SuperDL.cache_coordinator_client import CacheCoordinatorClient

def to_python_float(t:torch.Tensor)-> float:
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
  
def setup(config_file:str, devices: int, precision: Optional[str], resume: Union[bool, Path]) -> None:
    hparams = parse_args(config_file=config_file)
    
    precision = precision or get_default_supported_precision(training=True)
    
    if devices > 1:
        strategy = FSDPStrategy(state_dict_type="full",limit_all_gathers=True,cpu_offload=False,)
    else:
        strategy = "auto"

    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `lightning run model --help`
    logger = CSVLogger(save_dir=hparams.log_dir, name=hparams.dataset_name, flush_logs_every_n_steps=hparams.log_interval)

    fabric = L.Fabric(accelerator=hparams.accelerator,devices=hparams.devices, strategy=strategy, precision=precision, loggers=[logger])
    fabric.print(hparams)
    fabric.launch(main, resume=resume, hparams=hparams)
    main(fabric,resume=resume, hparams=hparams)


def main(fabric: L.Fabric, resume: Union[bool, Path],hparams:Namespace) -> None:
   
    # Set job ID
    hparams.job_id = os.getpid()

    # Register job with the cache coordinator service if 'use_super' is True
    if hparams.use_super:
        cache_coordinator_client = CacheCoordinatorClient(server_address=hparams.gprc_server_address)
        cache_coordinator_client.register_job(job_id= hparams.job_id, data_dir=hparams.data_dir, source_system='local')
    
    # Log hyperparameters
    fabric.loggers[0].log_hyperparams(vars(hparams))
    
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

    create_job_report(job_id =hparams.job_id,log_out_folder=fabric.loggers[0].log_dir)


def initialize_model(fabric: L.Fabric, arch: str) -> torch.nn.Module: 
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


def run_training(fabric: L.Fabric, state: dict, train_dataloader: DataLoader, val_dataloader: DataLoader, hparams:Namespace) -> None:
    
    train_logger = MLTrainingLogger(fabric_logger=fabric.loggers[0], log_interval=hparams.log_interval, prefix='train')
    val_logger = MLTrainingLogger(fabric_logger=fabric.loggers[0], log_interval=hparams.log_interval,prefix='val')

    best_prec1 = 0
    best_prec5 = 0
    best_loss = float('inf')  # Initialize with positive infinity

    for epoch in range(0, hparams.max_epochs):
        try:
            train_avg_loss,train_avg_top1,train_avg_top5 = train(fabric, state, train_dataloader, epoch, train_logger)
            state["scheduler"].step()
        except Exception as e:
            fabric.print("Error in training routine!")
            fabric.print(e)
            fabric.print(e.__class__.__name__)
            break
        
        if not hparams.train_only:
            try:
                val_avg_loss, val_avg_top1, val_avg_top5 = validate(fabric, state, val_dataloader, epoch, val_logger)
                best_prec1 = max(best_prec1, val_avg_top1)
                best_prec5 = max(best_prec5, val_avg_top5)
                best_loss = min(best_loss, val_avg_loss)
            except Exception as e:
                fabric.print("Error in validation routine!")
                fabric.print(e)
                fabric.print(e.__class__.__name__)
                break
        else:
            best_prec1 = max(best_prec1, train_avg_top1)
            best_prec5 = max(best_prec5, train_avg_top5)
            best_loss = min(best_loss, train_avg_loss)
        
        if hparams.always_save_checkpoint:
            fabric.save(state["model"].state_dict(), "renet_cnn.pt")
    

def train(fabric: L.Fabric, state: dict, train_dataloader: DataLoader,epoch: int, logger:MLTrainingLogger) -> None:
    
    model:torch.nn.Module = state["model"]
    optimizer:torch.optim.SGD = state["optimizer"]
    hparams = state["hparams"]
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    
    # switch to train mode
    fabric.print("training ...")
    model.train()
    
    total_batches = min(hparams.max_minibatches_per_epoch, len(train_dataloader)) if hparams.max_minibatches_per_epoch else len(train_dataloader)

    end = time.perf_counter()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        #no need to call `.to(device)` on the data, target
        batch_load_time = time.perf_counter() - end
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

        batch_compute_time = time.perf_counter() - compute_start
        # measure elapsed time
        total_batch_time = time.perf_counter() - end

        if hparams.max_minibatches_per_epoch and batch_idx >= hparams.max_minibatches_per_epoch-1:
            #end epoch early based on num_minibacthes that have been processed 
            break

        if batch_idx == len(train_dataloader) - 1:
            break

        logger.record_train_batch_metrics(
        epoch=epoch, batch_idx=batch_idx, num_samples=input.size(0),
        total_batch_time=total_batch_time, batch_load_time=batch_load_time,
        batch_compute_time=batch_compute_time, loss=losses.val,
        top1=top1.val, top5=top5.val,
        total_batches=total_batches, epoch_end=False, job_end=False
        )
        
        end = time.perf_counter()
    
    
    # Record metrics for the last batch and end of the epoch (and possibly job)
    logger.record_train_batch_metrics(
        epoch=epoch, batch_idx=batch_idx, num_samples=input.size(0),
        total_batch_time=total_batch_time, batch_load_time=batch_load_time,
        batch_compute_time=batch_compute_time, loss=losses.val,
        top1=top1.val, top5=top5.val,
        total_batches=total_batches, epoch_end=True, job_end=epoch == hparams.max_epochs-1
    )
    return losses.avg,top1.avg,top5.avg




def validate(fabric: L.Fabric, state: dict, val_dataloader: DataLoader, epoch: int, logger:MLTrainingLogger,) -> None:
    
    model:torch.nn.Module = state["model"]
    hparams = state["hparams"]
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    # switch to evaluate mode
    fabric.print("Validating ...")
    model.eval()
    total_batches = min(hparams.max_minibatches_per_epoch, len(val_dataloader)) if hparams.max_minibatches_per_epoch else len(val_dataloader)

    end = time.perf_counter()

    for batch_idx, (input, target) in enumerate(val_dataloader):

        batch_load_time = time.perf_counter() - end

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
        
        batch_compute_time = time.perf_counter() - compute_start
        total_batch_time = time.perf_counter() - end


        if hparams.max_minibatches_per_epoch and batch_idx >= hparams.max_minibatches_per_epoch-1:
            #end epoch early based on num_minibacthes that have been processed 
            break

        if batch_idx == len(val_dataloader) - 1:
            break

        logger.record_train_batch_metrics(
            epoch=epoch, batch_idx=batch_idx, num_samples=input.size(0),
            total_batch_time=total_batch_time, batch_load_time=batch_load_time,
            batch_compute_time=batch_compute_time, loss=to_python_float(loss.data),
            top1=to_python_float(prec1), top5=to_python_float(prec5),
            total_batches=total_batches, epoch_end=False, job_end=False)
 
        end = time.perf_counter()

        
    # Record metrics for the last batch and end of the epoch (and possibly job)
    logger.record_train_batch_metrics(
        epoch=epoch, batch_idx=batch_idx, num_samples=input.size(0),
        total_batch_time=total_batch_time, batch_load_time=batch_load_time,
        batch_compute_time=batch_compute_time, loss=to_python_float(loss.data),
        top1=to_python_float(prec1), top5=to_python_float(prec5),
        total_batches=total_batches, epoch_end=True, job_end=epoch == hparams.max_epochs-1
    )
    
    return losses.avg,top1.avg, top5.avg

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