import time
from lightning.fabric import Fabric
from typing import Any, Optional, Tuple, Union, Iterator
from pathlib import Path
from argparse import Namespace
from .utils import *
from .datasets import *
from .samplers import *
from torch.utils.data import DataLoader
from image_classification.logger import SUPERLogger
import torch.optim as optim
from datetime import datetime
from superdl.syncgrpc.client import SuperClient


def run_training(fabric: Fabric, model:torch.nn.Module, optimizer:optim.Optimizer, scheduler:optim.lr_scheduler.LRScheduler,
                train_dataloader: DataLoader, val_dataloader: DataLoader, hparams:Namespace,
                 logger:SUPERLogger, super_client:SuperClient = None) -> None:
    
    for epoch in range(hparams.workload.epochs):
        if hparams.workload.run_evaluate:
            
            fabric.print("validating..")
           
            total_batches = min(hparams.workload.max_minibatches_per_epoch, len(val_dataloader)) if hparams.workload.max_minibatches_per_epoch else len(val_dataloader)

            process_data(fabric=fabric,
                         dataloader=val_dataloader,
                         global_step=epoch * total_batches,
                         model=model,
                         optimizer=optimizer,
                         logger=logger,
                         epoch=epoch,
                         hparams=hparams,
                         is_training=False,
                         super_client=super_client,
                         total_batches=total_batches)
            
        if hparams.workload.run_training:
            fabric.print("training..")
            total_batches = min(hparams.workload.max_minibatches_per_epoch, len(train_dataloader)) if hparams.workload.max_minibatches_per_epoch else len(train_dataloader)

            process_data(fabric=fabric,
                         dataloader=train_dataloader,
                         global_step=epoch * total_batches,
                         model=model,
                         optimizer=optimizer,
                         logger=logger,
                         epoch=epoch,
                         hparams=hparams,
                         is_training=True,
                         super_client=super_client,
                         total_batches=total_batches)

    logger.job_end()
  


def process_data(fabric: Fabric, dataloader: DataLoader, 
                 global_step:int, model:torch.nn.Module, 
                 optimizer:torch.optim.SGD, logger:SUPERLogger, epoch, hparams:Namespace,
                 total_batches:int, is_training, super_client:SuperClient): 
    
    logger.epoch_start(epoch_length=total_batches,is_training=is_training)
    model.train(is_training)
    end = time.perf_counter()
    start_time = time.time()
    for iteration, (input, target, batch_id, cache_hit) in enumerate(dataloader):
        num_sampels = input.size(0)
        data_time = time.perf_counter() - end
        
        # Accumulate gradient x batches at a time
        is_accumulating = hparams.model.grad_acc_steps is not None and iteration % hparams.model.grad_acc_steps != 0

        if hparams.workload.profile:
            torch.cuda.synchronize()

        # Forward pass and loss calculation
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            output:torch.Tensor = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            if is_training:
                fabric.backward(loss) # .backward() accumulates when .zero_grad() wasn't called
        if not is_accumulating and is_training:
            # Step the optimizer after accumulation phase is over
            optimizer.step()
            optimizer.zero_grad()
        
        if hparams.workload.profile:
            torch.cuda.synchronize()    
          
        iteration_time = time.perf_counter()-end
        compute_time = iteration_time - data_time
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
  
        metrics_dict = logger.record_iteration_metrics(
            epoch=epoch,
            step=iteration,
            global_step = global_step,
            num_sampels=num_sampels,
            iteration_time=iteration_time,
            data_time=data_time,
            compute_time=compute_time,
            compute_ips=  calc_throughput_per_second(num_sampels,compute_time),
            total_ips=calc_throughput_per_second(num_sampels,iteration_time),
            loss = to_python_float(loss.detach()),
            top1=to_python_float(prec1),
            top5=to_python_float(prec5),
            batch_id=batch_id,
            is_training=is_training,
            cache_hit = cache_hit
            )
    
        if super_client is not None:
            metrics_dict['access_time'] = start_time
            metrics_dict['training_speed'] = logger.iteration_aggregator.compute_time.avg
            metrics_dict['cache_hit'] = cache_hit
           # super_client.share_job_metrics(dataset_id=dataloader.dataset.dataset_id, metrics=metrics_dict)

        global_step+=1

        #if (iteration + 1) % hparams.log_interval == 0:
        #    progress.display(iteration)

        if hparams.workload.max_minibatches_per_epoch and iteration >= hparams.workload.max_minibatches_per_epoch - 1:
            # end epoch early based on num_minibatches that have been processed 
            break

        if iteration == len(dataloader) - 1:
            break

        end = time.perf_counter()
        start_time = time.time()
    
    logger.epoch_end(epoch, is_training=is_training)



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



