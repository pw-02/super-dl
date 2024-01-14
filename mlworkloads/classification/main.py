import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union, Iterator
import hashlib
import redis
from torch import nn, optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from jsonargparse._namespace import Namespace

from lightning.fabric import Fabric
from superdl.syncgrpc.client import SuperClient
from image_classification.utils import *
from image_classification.datasets import *
from image_classification.samplers import *
from image_classification.training import *


def main(fabric: Fabric,hparams:Namespace) -> None:
    exp_start_time = time.time()
   
    # Prepare for training
    model, optimizer, scheduler, train_dataloader, val_dataloader, logger = prepare_for_training(
        fabric=fabric,hparams=hparams)
        
    logger.log_hyperparams(hparams)

    # Run training
    run_training(
        fabric=fabric,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        hparams=hparams,
        logger=logger
    )

    exp_duration = time.time() - exp_start_time

    create_job_report(hparams.workload.exp_name,logger.log_dir)

    fabric.print(f"Experiment ended. Duration: {exp_duration}")


def prepare_for_training(fabric: Fabric,hparams:Namespace):
    # Set seed
    if hparams.workload.seed is not None:
        fabric.seed_everything(hparams.workload.seed, workers=True)
    
    #Load model
    t0 = time.perf_counter()
    model = initialize_model(fabric, hparams.model.arch)
    fabric.print(f"Time to instantiate {hparams.model.arch} model: {time.perf_counter() - t0:.02f} seconds")
    fabric.print(f"Total parameters in {hparams.model.arch} model: {num_model_parameters(model):,}")

    #Initialize loss, optimizer and scheduler
    optimizer =  initialize_optimizer(optimizer_type = hparams.model.optimizer,  model_parameters=model.parameters(),learning_rate=hparams.model.lr, momentum=hparams.model.momentum, weight_decay=hparams.model.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30)) #TODO: Add support for other scheduler
     # call `setup` to prepare for model / optimizer for distributed training. The model is moved automatically to the right device.
    model, optimizer = fabric.setup(model,optimizer, move_to_device=True) 
    #Initialize data transformations
    transformations = initialize_transformations()

    # Initialize cache and super
    cache_client = initialize_cache_client(hparams.super_dl.use_cache, hparams.super_dl.cache_host, hparams.super_dl.cache_port)

    #Initialize datasets
    training_dataset, val_dataset = initialize_datasets(
        fabric=fabric, 
        dataloader_backend=hparams.data.dataloader_backend,
        transformations=transformations,
        data_dir=hparams.data.data_dir,
        use_s3=True if hparams.super_dl.source_system == "s3" else False,
        s3_bucket_name=hparams.data.s3_bucket_name,
        run_training= hparams.workload.run_training,
        run_evaluate= hparams.workload.run_evaluate,
        cache_client=cache_client
        )
    
    #Initialize super_client
    super_client = None
    if hparams.data.dataloader_backend == 'super':
        super_client = initialize_super_client(hparams.super_dl.server_address, hparams.job_id, hparams.super_dl.source_system, hparams.data.data_dir)                

    # Initialize Samplers
    train_sampler,val_sampler  = initialize_samplers(training_dataset, val_dataset,hparams.job_id, super_client,
                                                     shuffle=hparams.data.shuffle,batch_size=hparams.data.batch_size,
                                                     drop_last=hparams.data.drop_last,prefetch_lookahead=hparams.super_dl.prefetch_lookahead)
    # Initialize DataLoaders
    if hparams.workload.run_training:
        train_dataloader= DataLoader(training_dataset, sampler=train_sampler, batch_size=None, num_workers=hparams.pytorch.workers)
        train_dataloader = fabric.setup_dataloaders(train_dataloader)

    if hparams.workload.run_evaluate:
        val_dataloader= DataLoader(val_dataset, sampler=val_sampler, batch_size=None, num_workers=hparams.pytorch.workers)
        val_dataloader = fabric.setup_dataloaders(val_dataloader)

    #Initialize logger
    logger = SUPERLogger( fabric=fabric, root_dir=hparams.workload.log_dir,
                          flush_logs_every_n_steps=hparams.workload.flush_logs_every_n_steps,
                          print_freq= hparams.workload.print_freq,
                          exp_name=hparams.workload.exp_name)
   
    
    return model, optimizer, scheduler, train_dataloader, val_dataloader, logger


def initialize_model(fabric: Fabric, arch: str) -> nn.Module: 
    with fabric.init_module(empty_init=True): #model is instantiated with randomly initialized weights by default.
        model: nn.Module = models.get_model(arch)
    return model

def initialize_optimizer(optimizer_type:str, model_parameters:Iterator[nn.Parameter], learning_rate, momentum, weight_decay):
    if optimizer_type == "sgd":
        optimizer = optim.SGD(params=model_parameters, 
                              lr=learning_rate, 
                              momentum=momentum, 
                              weight_decay=weight_decay)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(params=model_parameters, 
                              lr=learning_rate,
                              momentum=momentum, 
                              weight_decay=weight_decay)
    return optimizer

def initialize_cache_client(use_cache: bool, cache_host: str, cache_port: int):
    return redis.StrictRedis(host=cache_host, port=cache_port) if use_cache else None


def initialize_super_client(server_address: str, job_id, data_source_system, data_dir):
    
    dataset_id = hashlib.sha256(f"{data_source_system}_{data_dir}".encode()).hexdigest()
    super_client = SuperClient(server_address)
    super_client.register_new_job(job_id=job_id,dataset_id=dataset_id, data_source_system=data_source_system, data_dir=data_dir)
    return super_client
    

def initialize_samplers(training_dataset, validation_dataset, job_id,super_client,shuffle, batch_size, drop_last,prefetch_lookahead):
    train_sampler = None
    val_sampler = None

    if training_dataset is not None:
        train_sampler = SUPERSampler(
        dataset=training_dataset,
        job_id=job_id,
        super_client=super_client,
        shuffle=shuffle,
        prefix='train',
        seed=1,
        batch_size=batch_size,
        drop_last=drop_last,
        prefetch_lookahead=prefetch_lookahead
        )
    if validation_dataset is not None:
        val_sampler = SUPERSampler(
        dataset=validation_dataset,
        job_id=job_id,
        super_client=super_client,
        shuffle=False,
        prefix='val',
        seed=1,
        batch_size=batch_size,
        drop_last=drop_last,
        prefetch_lookahead=prefetch_lookahead
        )
    return train_sampler, val_sampler
        

def initialize_datasets(fabric: Fabric, 
                        dataloader_backend:str, 
                        transformations: transforms.Compose,
                        data_dir:str,
                        use_s3:bool,
                        s3_bucket_name:str,
                        run_training:bool,
                        run_evaluate:bool,
                        cache_client=None,
                        ):
    
    train_data = None
    val_data = None

    if dataloader_backend == "super":
        if run_training:     
           train_data = SUPERDataset(
                fabric=fabric,
                prefix='train',
                data_dir=data_dir,
                transform=transformations,
                cache_client=cache_client,
                use_s3=use_s3,
                s3_bucket_name=s3_bucket_name
                )
        
        if run_evaluate:     
            val_data = SUPERDataset(
                fabric=fabric,
                prefix='val',
                data_dir=data_dir,
                transform=transformations,
                cache_client=cache_client,
                use_s3=use_s3,
                s3_bucket_name=s3_bucket_name
                )
    return train_data, val_data

def initialize_transformations() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.ToTensor(), normalize])
    return transformations

if __name__ == "__main__":
    pass