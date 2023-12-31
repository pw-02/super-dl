from lightning.pytorch.loggers import CSVLogger
from misc.utils import AverageMeter, ProgressMeter
class MLTrainingLogger:
    def __init__(self, fabric_logger: CSVLogger, log_interval: int ):
        self.fabric_logger = fabric_logger
        self.init_meters()
        self.log_interval = log_interval
        self.step_count = 0

    def init_meters(self):
        self.job_total_samples:int = 0
        self.job_total_batches:int = 0
        self.job_total_epochs:int = 0
        self.job_total_duration: float = 0.0
        self.job_total_dataloading_time: float = 0.0
        self.job_total_compute_time: float = 0.0
        self.job_samples_per_sec: float = 0.0
        self.job_batches_per_sec: float = 0.0
                    
        self.job_epochs_per_sec: float = 0.0
        self.job_best_prec1: float = 0.0
        self.job_best_prec5: float = 0.0
        self.job_best_loss: float = float('inf')  # Initialize with positive infinity

        self.epoch_batch_load_times = AverageMeter("batch", ":6.3f")
        self.epoch_batch_times = AverageMeter("data", ":6.3f")
        self.epoch_batch_compute_times = AverageMeter("compute", ":6.3f")
        self.epoch_batch_losses = AverageMeter("Loss", ":.4e")
        self.epoch_batch_top1_acc = AverageMeter("Acc@1", ":6.2f")
        self.epoch_batch_top5_acc = AverageMeter("Acc@5", ":6.2f")
        self.epoch_num_samples = AverageMeter("Samples", ":6.3f")

    def reset_epoch_meters(self):
        meters_to_reset = [
            self.epoch_batch_load_times,
            self.epoch_batch_compute_times,
            self.epoch_batch_times,
            self.epoch_batch_losses,
            self.epoch_batch_top1_acc,
            self.epoch_batch_top5_acc,
            self.epoch_num_samples

        ]
        
        for meter in meters_to_reset:
            meter.reset()
    
    def should_log_metrics(self, batch_idx: int) -> bool:
        """Check if metrics should be logged based on the log interval."""
        return (batch_idx == 0) or (batch_idx % self.log_interval == 0)

    def record_train_batch_metrics(self, epoch, batch_idx,num_samples,total_batch_time,batch_load_time, 
                                   batch_compute_time,loss, top1, top5, 
                                   total_batches,epoch_end = False, job_end = False):

        self.epoch_batch_times.update(total_batch_time)
        self.epoch_batch_load_times.update(batch_load_time)
        self.epoch_batch_compute_times.update(batch_compute_time)
        self.epoch_batch_losses.update(loss)
        self.epoch_batch_top5_acc.update(top5) 
        self.epoch_batch_top1_acc.update(top1)
        self.epoch_num_samples.update(num_samples)
        self.step_count +=1

        if job_end or epoch_end:
            metrics ={}
            metrics.update(self.gen_batch_metrics_for_log(epoch=epoch,batch_idx=batch_idx))
            metrics.update(self.gen_epoch_metrics_for_log())

            self.update_job_totals()

            if job_end:
                metrics.update(self.gen_job_metrics_for_log())
            
            self.fabric_logger.log_metrics(metrics, self.step_count)
            self._display_progress(epoch, batch_idx, total_batches,epoch_end, job_end)
            
            self.reset_epoch_meters()

        elif self.should_log_metrics(batch_idx):
            self._display_progress(epoch, batch_idx, total_batches, epoch_end, job_end)
            self.fabric_logger.log_metrics(self.gen_batch_metrics_for_log(epoch=epoch,batch_idx=batch_idx), self.step_count)

    
    def gen_batch_metrics_for_log(self, epoch,batch_idx ):
        return {
             "epoch_idx": epoch,
             "batch_idx": batch_idx,
             "batch/num_samples": self.epoch_num_samples.val,
             "batch/total_time": self.epoch_batch_times.val,
             "batch/load_time": self.epoch_batch_load_times.val,
             "batch/compute_time": self.epoch_batch_compute_times.val,
             "batch/throughput/samples_per_sec": self.epoch_num_samples.val / self.epoch_batch_times.val,
             "batch/loss": self.epoch_batch_losses.avg,
             "batch/acc@1": self.epoch_batch_top1_acc.avg,
             "batch/acc@5": self.epoch_batch_top5_acc.avg,
             }
        
    def gen_epoch_metrics_for_log(self):
        return {
                "epoch/total_batches": self.epoch_batch_times.count,
                "epoch/total_samples": self.epoch_num_samples.val,
                "epoch/epoch_time": self.epoch_batch_times.sum,
                "epoch/dataloading_time": self.epoch_batch_load_times.sum,
                "epoch/compute_time": self.epoch_batch_compute_times.sum,
                "epoch/throughput/samples_per_sec": self.epoch_num_samples.val / self.epoch_batch_times.sum,
                "epoch/throughput/batches_per_sec": self.epoch_batch_times.count / self.epoch_batch_times.sum,
                "epoch/loss": self.epoch_batch_losses.avg,
                "epoch/acc@1": self.epoch_batch_top1_acc,
                "epoch/acc@5": self.epoch_batch_top5_acc.avg,
                }
    
    def gen_job_metrics_for_log(self):
        return {
                 "job/total_samples":self.job_total_samples,
                 "job/total_batches": self.job_total_batches,
                 "job/total_epochs":  self.job_total_epochs,
                 "job/job_time": self.job_total_duration,
                 "job/dataloading_time": self.job_total_dataloading_time,
                 "job/compute_time": self.job_total_compute_time,
                 "job/throughput/samples_per_sec": self.job_samples_per_sec,
                 "job/throughput/batches_per_sec": self.job_batches_per_sec,
                 "job/throughput/epochs_per_sec":  self.job_epochs_per_sec,
                  "job/best_acc1": self.job_best_prec1,
                  "job/best_acc5":  self.job_best_prec5,
                  "job/loss": self.job_best_loss,
                  }

    def update_job_totals(self):
        self.job_total_samples += self.epoch_num_samples.val
        self.job_total_batches += self.epoch_batch_times.count
        self.job_total_epochs +=1
        self.job_total_duration += self.epoch_batch_times.sum
        self.job_total_dataloading_time += self.epoch_batch_load_times.sum
        self.job_total_compute_time += self.epoch_batch_compute_times.sum
        self.job_samples_per_sec += self.job_total_samples / self.epoch_batch_times.sum
        self.job_batches_per_sec += self.job_total_batches / self.epoch_batch_times.sum
        self.job_epochs_per_sec += self.job_total_epochs / self.epoch_batch_times.sum
        self.job_best_prec1 = max(self.job_best_prec1, self.epoch_batch_top1_acc.avg)
        self.job_best_prec5 = max(self.job_best_prec5, self.epoch_batch_top5_acc.avg)
        self.job_best_loss = min(self.job_best_loss, self.epoch_batch_losses.avg)

    def _display_progress(self, epoch, batch_idx, total_batches, end_of_epoch, end_of_job ):

        def print_seperator_line():
            print('-' * 140)

        decimal_places = 4
        # Display progress information
        progress_info = {
        "Epoch": f"[{epoch}]",
        "Batch": f"{batch_idx}/{total_batches}",
        "TotalTime": round(self.epoch_batch_times.val, decimal_places),
        "DataLoadTime": round(self.epoch_batch_load_times.val, decimal_places),
        "ComputeTime": round(self.epoch_batch_compute_times.val, decimal_places),
        "Loss": round(self.epoch_batch_losses.val, decimal_places),
        "Acc@1": round(self.epoch_batch_top1_acc.val, decimal_places)
        }
        
        print(", ".join(f"{key}: {value}" for key, value in progress_info.items()))

        if end_of_epoch:
            print_seperator_line()
            print(f"EPOCH [{epoch}] SUMMARY:")
            epoch_info = {
                "TotalTime": round(self.epoch_batch_times.sum, decimal_places),
                "DataLoadTime": round(self.epoch_batch_load_times.sum, decimal_places),
                "ComputeTime": round(self.epoch_batch_compute_times.sum, decimal_places),
                "Loss": round(self.epoch_batch_losses.avg, decimal_places),
                "AvgAcc@1": round(self.epoch_batch_top1_acc.avg, decimal_places),
                "AvgAcc@5": round(self.epoch_batch_top5_acc.avg, decimal_places)

            }

            print(", ".join(f"{key}: {value}" for key, value in epoch_info.items()))
            print_seperator_line()
        
        if end_of_job:
            print_seperator_line()
            print(f"JOB [{epoch}] SUMMARY:")
            job_info = {
                "TotalEpochs": round(self.job_total_epochs, decimal_places),
                "TotalBatches": round(self.job_total_batches, decimal_places),
                "TotalTime": round(self.job_total_duration, decimal_places),
                "DataLoadTime": round(self.job_total_dataloading_time, decimal_places),
                "ComputeTime": round(self.job_total_compute_time, decimal_places),
                "Loss": round(self.job_best_loss, decimal_places),
                "AvgAcc@1": round(self.job_best_prec1, decimal_places),
                "AvgAcc@5": round(self.job_best_prec5, decimal_places),
            }

            print(", ".join(f"{key}: {value}" for key, value in job_info.items()))
            print_seperator_line()



    
    def _log_epoch_metrics(self, log_data, step_count):
        # Log epoch-level metrics
        metrics_info = {
            "Epoch": log_data['epoch'],
            "Loss": f"{log_data['epoch/loss']:.4e}",
            "Acc@1": f"{log_data['epoch/acc@1']:.2f}",
            "Acc@5": f"{log_data['epoch/acc@5']:.2f}"
        }
        summary = "Epoch {Epoch} summary - Loss: {Loss}, Acc@1: {Acc@1}, Acc@5: {Acc@5}".format(**metrics_info)
        print(summary)