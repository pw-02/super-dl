import time
from lightning.fabric import Fabric
from typing import Any, Optional, Tuple, Union, Iterator
from pathlib import Path
from argparse import Namespace
from torchvision import models, transforms
from torch import nn
from image_classification.utils import *
from image_classification.datasets import *
from image_classification.samplers import *
from torch import optim
from torch.utils.data import DataLoader
from image_classification.training import *
from super_dl.cache_coordinator_client import CacheCoordinatorClient

def main(fabric: Fabric,hparams:Namespace) -> None:
    exp_start_time = time.time()
   
    # Prepare for training
    model, optimizer, scheduler, train_dataloader, val_dataloader, logger = prepare_for_training(
        fabric=fabric,hparams=hparams)

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

    if fabric.rank_zero_first(True):
        logger.create_job_report()

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

    # Initialize loss, optimizer and scheduler
    optimizer =  initialize_optimizer(model_hparams=hparams.model, model_parameters=model.parameters())
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30)) #TODO: Add support for other scheduler

     # call `setup` to prepare for model / optimizer for distributed training. The model is moved automatically to the right device.
    model, optimizer = fabric.setup(model,optimizer, move_to_device=True) 
    #Initialize data transformations
    transformations = initialize_transformations()

    #Initialize DataLoaders
    train_dataloader = None
    val_dataloader = None
    if hparams.workload.run_training:
        train_dataloader = initialize_dataloader(fabric=fabric,hparams=hparams, transformations=transformations, prefix='train')
        train_dataloader = fabric.setup_dataloaders(train_dataloader)
    if hparams.workload.run_evaluate:
        val_dataloader = initialize_dataloader(fabric=fabric,hparams=hparams, transformations=transformations, prefix='val')
        val_dataloader = fabric.setup_dataloaders(val_dataloader)

    #Initialize logger
    logger = SUPERLogger(root_dir=hparams.workload.log_dir,
                          rank=fabric.local_rank, 
                          flush_logs_every_n_steps=hparams.workload.flush_logs_every_n_steps,
                          print_freq= hparams.workload.print_freq,
                          exp_name=hparams.workload.exp_name)
    logger.log_hyperparams(hparams)
    
    return model, optimizer, scheduler, train_dataloader, val_dataloader, logger


def initialize_model(fabric: Fabric, arch: str) -> nn.Module: 
    with fabric.init_module(empty_init=True): #model is instantiated with randomly initialized weights by default.
        model: nn.Module = models.get_model(arch)
    return model

def initialize_optimizer(model_hparams:Namespace, model_parameters:Iterator[nn.Parameter]):
    if model_hparams.optimizer == "sgd":
        optimizer = optim.SGD(params=model_parameters, 
                              lr=model_hparams.lr, 
                              momentum=model_hparams.momentum, 
                              weight_decay=model_hparams.weight_decay)
    elif model_hparams.optimizer == "rmsprop":
        optimizer = optim.RMSprop(params=model_parameters, 
                              lr=model_hparams.lr,
                              momentum=model_hparams.momentum, 
                              weight_decay=model_hparams.weight_decay)
    return optimizer

def initialize_dataloader(
    fabric: Fabric,
    hparams: Namespace,
    transformations: transforms.Compose,
    prefix: str,
) -> DataLoader:
    
    dataloader = None  # Initialize to None for clarity

    if hparams.data.dataloader_backend == "super":
        cache_client:CacheCoordinatorClient = None
        super_client:CacheCoordinatorClient = None
       
        if hparams.super_dl.use_coordinator:
            pass
            # Create connection to the super client
            # cache_coordinator_client = CacheCoordinatorClient(server_address=hparams.gprc_server_address)
            # cache_coordinator_client.register_job(job_id=hparams.job_id, data_dir=hparams.data_dir, source_system='local')

        if hparams.super_dl.use_cache:
            pass

        if hparams.super_dl.mode == "local":
            dataset = SUPERLocalDataset(
                fabric=fabric,
                prefix=prefix,
                data_dir=hparams.data.data_dir,
                transform=transformations,
                cache_client=cache_client,
                super_client=super_client
            )
        elif hparams.super_dl.mode == "s3":
            dataset = SUPERS3Dataset(
                fabric=fabric,
                prefix=prefix,
                lambda_function_name=hparams.super_dls3_lambda_name,
                data_dir=hparams.data.data_dir,
                transform=transformations,
                cache_client=cache_client,
                super_client=super_client
            )
        else:
            print("Invalid super_dl mode selected")
            exit(1)
        
        sampler = SUPERSampler(
            dataset=dataset,
            job_id=hparams.job_id,
            super_client=super_client,
            shuffle=hparams.data.shuffle,
            seed=4,
            batch_size=hparams.data.batch_size,
            drop_last=hparams.data.drop_last,
            prefetch_look_ahead=hparams.super_dl.prefetch_lookahead
        )
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=None, num_workers=hparams.pytorch.workers)

    elif hparams.data_backend == "pytorch":
        # Handle pytorch specific dataset initialization if needed
        pass
    else:
        print("Bad databackend picked")
        exit(1)
    return dataloader

def initialize_transformations() -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transformations = transforms.Compose([transforms.ToTensor(), normalize])
    return transformations

if __name__ == "__main__":
    pass