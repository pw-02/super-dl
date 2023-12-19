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
from torchmetrics.classification import Accuracy
from pytorch.custom_dataset import SUPERVisionDataset
from pytorch.custom_batch_sampler import SimpleBatchSampler
import torch.nn.functional as F

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    
def setup(devices: int = 1, precision: Optional[str] = None, resume: Union[bool, Path] = False) -> None:

    hparams = parse_args(default_config_file='mlworkloads/cfgs/resnet18.yaml')
    precision = precision or get_default_supported_precision(training=True)
    
    if devices > 1:
        strategy = FSDPStrategy(
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    # Create the Lightning Fabric object. The parameters like accelerator, strategy, devices etc. will be proided
    # by the command line. See all options: `lightning run model --help`
    
    logger = CSVLogger(hparams.log_dir, hparams.dataset_name, flush_logs_every_n_steps=hparams.log_interval)
    fabric = L.Fabric(accelerator="gpu",devices=devices, strategy=strategy, precision=precision, loggers=[logger])
    fabric.print(hparams)
    fabric.launch(main, resume=resume, hparams=hparams)

def main(fabric: L.Fabric, resume: Union[bool, Path],hparams) -> None:

    from torchvision import models, transforms

    fabric.loggers[0].log_hyperparams(vars(hparams))
    
    if fabric.global_rank == 0 and hparams.always_save_checkpoint:
        hparams.save_dir.mkdir(parents=True, exist_ok=True)
    
    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP) # instead of torch.manual_seed(...)

    fabric.print(f"Loading {hparams.arch} model")
    t0 = time.perf_counter()
    
    with fabric.init_module(empty_init=True):
        model:torch.nn.Module = models.get_model(hparams.arch) #model is instantiated with randomly initialized weights by default.
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters in {hparams.arch}: {num_parameters(model):,}")
    optimizer = torch.optim.SGD(model.parameters(), lr=hparams.lr, momentum=hparams.momentum, weight_decay=hparams.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.1 ** (epoch // 30))

    # call `setup` to prepare for model / optimizer for distributed training.
    # the model is moved automatically to the right device.
    model, optimizer = fabric.setup(model,optimizer)  

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transformations = transforms.Compose(
        [#transforms.Resize(256), 
         #transforms.CenterCrop(224), 
         transforms.ToTensor(), normalize]
    )
 
    train_data = SUPERVisionDataset(source_system=hparams.source_system,cache_host=hparams.cache_host,
                                    data_dir=hparams.data_dir,prefix='train', transform=transformations)
    val_data = SUPERVisionDataset(source_system=hparams.source_system,cache_host=hparams.cache_host,
                                    data_dir=hparams.data_dir,prefix='val',transform=transformations)
    
    train_sampler = SimpleBatchSampler(dataset_size=len(train_data),batch_size=hparams.batch_size)
    val_sampler = SimpleBatchSampler(dataset_size=len(val_data),batch_size=hparams.batch_size)
    
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=None, num_workers=hparams.num_workers)
    val_dataloader = DataLoader(val_data,sampler=val_sampler, batch_size=None, num_workers=hparams.num_workers)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)
    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    '''
    train_time = time.perf_counter()
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")
    '''
    # use torchmetrics instead of manually computing the accuracy
    #test_acc:Accuracy = Accuracy(task="multiclass", num_classes=10).to(fabric.device)

    job_dataloading_time = AverageMeter("Data", ":6.3f")
    job_compute_time = AverageMeter("Compute", ":6.3f")
    job_num_samples = AverageMeter("Samples", ":6.3f")
    job_num_batches = AverageMeter("batches", ":6.3f")
    job_epoch_times = AverageMeter("Epochs", ":6.3f")
    best_prec1 = 0
    best_prec5 = 0
    best_loss = 0

    for epoch in range(1, hparams.max_epochs +1):

        epoch_train_metrics = train(fabric, state, train_dataloader, epoch)

        scheduler.step()

        if hparams.no_eval:
            prec1, prec5, loss = 0,0,0
        else:
            prec1, prec5, loss =  validate(fabric=fabric, model=model,val_dataloader=val_dataloader, hparams=hparams)
            best_prec1 = max( best_prec1,prec1)
            best_prec5 = max( best_prec5,prec5)
            best_loss = min (best_loss, loss)

        epoch_train_metrics['epoch/acc@1'] = prec1
        epoch_train_metrics['epoch/acc@5'] = prec5
        epoch_train_metrics['epoch/loss'] = loss

        job_dataloading_time.update(epoch_train_metrics["epoch/dataloading_time"])
        job_compute_time.update(epoch_train_metrics["epoch/compute_time"])
        job_num_samples.update(epoch_train_metrics["epoch/total_samples"])
        job_num_batches.update(job_num_batches.val + epoch_train_metrics["epoch/total_batches"])
        job_epoch_times.update(epoch_train_metrics["epoch/epoch_time"])
        
        if epoch != hparams.max_epochs:
             fabric.loggers[0].log_metrics(epoch_train_metrics, state["step_count"])

        # When using distributed training, use `fabric.save` to ensure the current process is allowed to save a checkpoint     
        if hparams.always_save_checkpoint:
             fabric.save(model.state_dict(), "renet_cnn.pt")
    
    job_metrics = {
                 "job/total_samples": job_num_samples.sum,
                 "job/total_batches": job_num_batches.sum,
                 "job/total_epochs": job_epoch_times.count,
                 "job/job_time": job_epoch_times.sum,
                 "job/dataloading_time": job_dataloading_time.sum,
                 "job/compute_time": job_compute_time.sum,
                 "job/throughput/samples_per_sec": job_num_samples.sum / job_epoch_times.sum,
                 "job/throughput/bathces_per_sec": job_num_batches.sum / job_epoch_times.sum,
                 "job/throughput/epochs_per_sec": job_epoch_times.count / job_epoch_times.sum,
                  "job/best_acc1": best_prec1,
                  "job/best_acc5": best_prec5,
                  "job/loss": best_loss,
                  }
    epoch_train_metrics.update(job_metrics)
    fabric.loggers[0].log_metrics(epoch_train_metrics,state["step_count"])



'''
model: This is your neural network model, which takes input data (inputs) and produces output predictions (outputs).
criterion: This is your loss function, which measures the difference between the model's predictions and the true targets.
optimizer: This is the optimization algorithm used to update the model's parameters based on the computed gradients.
The forward pass involves feeding the input data (inputs) through the neural network (model) to obtain predictions (outputs).
The loss is then computed by comparing the predictions to the true targets (targets) using the specified loss function (criterion).
The backward pass is where the gradients of the loss with respect to the model parameters are computed.
optimizer.zero_grad() is called to zero out the gradients of all the model parameters. This is essential to prevent gradient accumulation from previous iterations.
loss.backward() computes the gradients of the loss with respect to each model parameter using the chain rule of calculus.
After the backward pass, the optimizer (optimizer.step()) is called to update the model parameters based on the computed gradients.
The optimizer uses an optimization algorithm (in this case, stochastic gradient descent - SGD) to adjust the parameters in the direction that reduces the loss.
Gradients represent the partial derivatives of the loss with respect to each model parameter. They indicate the direction and magnitude of the steepest increase of the loss.
During the backward pass, gradients are computed using backpropagation, and they guide the optimizer in adjusting the model parameters to minimize the loss.
'''

def train(fabric: L.Fabric, state: dict, train_dataloader: DataLoader,epoch: int) -> None:
   
    model:torch.nn.Module = state["model"]
    optimizer:torch.optim.SGD = state["optimizer"]
    hparams = state["hparams"]

    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    compute_time = AverageMeter("compute", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_dataloader), [batch_time, data_time, compute_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    fabric.print("training ...")
    model.train()

    end = time.perf_counter()

    for batch_idx, (input, target) in enumerate(train_dataloader):
        # NOTE: no need to call `.to(device)` on the data, target
        data_time.update(time.perf_counter() - end)

        #-----------------Stop data, start compute------#
        if hparams.data_profile:
            torch.cuda.synchronize() # Synchronize before timing GPU operations  
        
        compute_start = time.perf_counter()

        # Forward pass
        output = model(input)
        loss = F.cross_entropy(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(to_python_float(loss.data), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        
        #Backward pass
        optimizer.zero_grad() # Zero the gradients
        fabric.backward(loss)  # instead of loss.backward()  #Computes gradients
        optimizer.step()  # Update model parameters
        
        torch.cuda.synchronize()

        compute_time.update(time.perf_counter() - compute_start)
        # measure elapsed time
        batch_time.update(time.perf_counter() - end)

        state["step_count"] +=1
        
        log_data = {
             "epoch": epoch,
             "batch_idx": batch_idx,
             "batch/num_samples": input.size(0),
             "batch/batch_time": batch_time.val,
             "batch/dataloading_time": data_time.val,
             "batch/compute_time": compute_time.val,
             "batch/throughput/samples_per_sec": input.size(0) / batch_time.val,
             "batch/loss": losses.avg,
             "batch/acc@1": top1.avg,
             "batch/acc@5": top5.avg,
             }
        
        if batch_idx + 1 == hparams.num_minibatches and not hparams.full_epoch or batch_idx == len(train_dataloader):
            progress.display(batch_idx)
            
            epoch_metrics = {
                 "epoch/total_batches": data_time.count,
                 "epoch/total_samples": len(train_dataloader.dataset),
                 "epoch/epoch_time": batch_time.sum,
                 "epoch/dataloading_time": data_time.sum,
                 "epoch/compute_time": compute_time.sum,
                 "epoch/throughput/samples_per_sec": len(train_dataloader.dataset)/ batch_time.sum,
                 "epoch/loss": losses.avg,
                 "epoch/acc@1": top1.avg,
                 "epoch/acc@5": top5.avg,
                 }
            
            log_data.update(epoch_metrics)


        elif (batch_idx == 0) or ((batch_idx + 1) % hparams.log_interval == 0):       
                progress.display(batch_idx)
                fabric.loggers[0].log_metrics(log_data,state["step_count"])
        
        end = time.perf_counter()

        if batch_idx+1 == hparams.num_minibatches and not hparams.full_epoch:
            break

    return log_data


def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, hparams) -> torch.Tensor:
    
    batch_time = AverageMeter("batch", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    
    # switch to evaluate mode
    fabric.print("Validating ...")
    model.eval()

    end = time.time()

    test_loss = 0
    for batch_idx, (input, target) in enumerate(val_dataloader):
        # compute output
        with torch.no_grad():
            output = model(input)
            loss = F.cross_entropy(output, target)
        
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        
        losses.update(to_python_float(loss), input.size(0))
        top1.update(to_python_float(prec1), input.size(0))
        top5.update(to_python_float(prec5), input.size(0))
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_idx+1 == hparams.num_minibatches and not hparams.full_epoch:
            break
    
    fabric.print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return top1.avg, top5.avg, losses.avg
    
    # all_gather is used to aggregated the value across processes
    #test_loss = fabric.all_gather(test_loss).sum() / len(val_dataloader.dataset)
    #print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: ({100 * test_acc.compute():.0f}%)\n")

def accuracy(output, target, topk=(1,)):
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

    from jsonargparse import CLI
    CLI(setup)