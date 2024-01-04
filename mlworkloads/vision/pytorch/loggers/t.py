import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)

class SuperLogger:
    
    def __init__(self, root_dir:_PATH, job_id:int, prefix:str, rank:int, flush_logs_every_n_steps:int = 50, name: str = "logs",version: Optional[Union[int, str]] = None):
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)
        self._flush_logs_every_n_steps = flush_logs_every_n_steps

    @property
    def name(self) -> str:
        """Gets the name of the experiment.

        Returns:
            The name of the experiment.

        """
        return self._name
    
    @property
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def root_dir(self) -> str:
        """Gets the save directory where the versioned CSV experiments are saved."""
        return self._root_dir
    
    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self.name, version)
    

    
    
    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self.name)
        if not _is_dir(self._fs, versions_root, strict=True):
            log.warning("Missing logger folder: %s", versions_root)
            return 0
        existing_versions = []
        for d in self._fs.listdir(versions_root):
            full_path = d["name"]
            name = os.path.basename(full_path)
            if _is_dir(self._fs, full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1
    

  

    
    

class MLTrainingLogger:
    def __init__(self, fabric_logger: CSVLogger, log_interval: int, prefix:str ):
        self.fabric_logger = fabric_logger
        self.init_meters()
        self.log_interval = log_interval
        self.step_count = 0
        self.prefix = prefix

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
            metrics.update(self.gen_epoch_metrics_for_log(epoch=epoch))

            self.update_job_totals()

            if job_end:
                metrics.update(self.gen_job_metrics_for_log())
            
            self.fabric_logger.log_metrics(metrics, self.step_count)
            self._display_progress(epoch, batch_idx, total_batches,epoch_end, job_end)
            
            self.reset_epoch_meters()

        elif self.should_log_metrics(batch_idx):
            self._display_progress(epoch, batch_idx, total_batches, epoch_end, job_end)
            self.fabric_logger.log_metrics(self.gen_batch_metrics_for_log(epoch=epoch,batch_idx=batch_idx), self.step_count)

    
    def gen_batch_metrics_for_log(self, epoch, batch_idx):
        return {
            f"{self.prefix}-epoch_idx": epoch,
            f"{self.prefix}-batch_idx": batch_idx,
            f"{self.prefix}-batch/num_samples": self.epoch_num_samples.val,
            f"{self.prefix}-batch/total_time": self.epoch_batch_times.val,
            f"{self.prefix}-batch/load_time": self.epoch_batch_load_times.val,
            f"{self.prefix}-batch/compute_time": self.epoch_batch_compute_times.val,
            f"{self.prefix}-batch/throughput/samples_per_sec": self.epoch_num_samples.val / self.epoch_batch_times.val,
            f"{self.prefix}-batch/loss": self.epoch_batch_losses.avg,
            f"{self.prefix}-batch/acc@1": self.epoch_batch_top1_acc.avg,
            f"{self.prefix}-batch/acc@5": self.epoch_batch_top5_acc.avg,
        }
        
    def gen_epoch_metrics_for_log(self,epoch):
        return {
            f"{self.prefix}-epoch/epoch_indx": epoch,
            f"{self.prefix}-epoch/total_batches": self.epoch_batch_times.count,
            f"{self.prefix}-epoch/total_samples": self.epoch_num_samples.val,
            f"{self.prefix}-epoch/epoch_time": self.epoch_batch_times.sum,
            f"{self.prefix}-epoch/dataloading_time": self.epoch_batch_load_times.sum,
            f"{self.prefix}-epoch/compute_time": self.epoch_batch_compute_times.sum,
            f"{self.prefix}-epoch/throughput/samples_per_sec": self.epoch_num_samples.val / self.epoch_batch_times.sum,
            f"{self.prefix}-epoch/throughput/batches_per_sec": self.epoch_batch_times.count / self.epoch_batch_times.sum,
            f"{self.prefix}-epoch/loss": self.epoch_batch_losses.avg,
            f"{self.prefix}-epoch/acc@1": self.epoch_batch_top1_acc.avg,
            f"{self.prefix}-epoch/acc@5": self.epoch_batch_top5_acc.avg,
        }
    
    def gen_job_metrics_for_log(self):
        return {
            f"{self.prefix}-job/total_samples": self.job_total_samples,
            f"{self.prefix}-job/total_batches": self.job_total_batches,
            f"{self.prefix}-job/total_epochs": self.job_total_epochs,
            f"{self.prefix}-job/job_time": self.job_total_duration,
            f"{self.prefix}-job/dataloading_time": self.job_total_dataloading_time,
            f"{self.prefix}-job/compute_time": self.job_total_compute_time,
            f"{self.prefix}-job/throughput/samples_per_sec": self.job_samples_per_sec,
            f"{self.prefix}-job/throughput/batches_per_sec": self.job_batches_per_sec,
            f"{self.prefix}-job/throughput/epochs_per_sec": self.job_epochs_per_sec,
            f"{self.prefix}-job/best_acc1": self.job_best_prec1,
            f"{self.prefix}-job/best_acc5": self.job_best_prec5,
            f"{self.prefix}-job/loss": self.job_best_loss,
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
            print('-' * 100)

        decimal_places = 4
        # Display progress information
        progress_info = {
        "Epoch": f"[{epoch}]",
        "Batch": f"{batch_idx}/{total_batches-1}",
        "TotalTime": round(self.epoch_batch_times.val, decimal_places),
        "DataLoadTime": round(self.epoch_batch_load_times.val, decimal_places),
        "ComputeTime": round(self.epoch_batch_compute_times.val, decimal_places),
        "Loss": round(self.epoch_batch_losses.val, decimal_places),
        "Acc@1": round(self.epoch_batch_top1_acc.val, decimal_places)
        }
        
        print(", ".join(f"{key}: {value}" for key, value in progress_info.items()))

        if end_of_epoch:
            print_seperator_line()
            print(f"{self.prefix.upper()}-EPOCH {epoch} SUMMARY:")
            epoch_info = {
                "TotalBatches": self.epoch_batch_times.count,
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
            print(f"{self.prefix.upper()}-JOB SUMMARY:")
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
    
def create_job_report(job_id, log_out_folder):
        import pandas as pd
        import os
        import yaml

        #Check if a file called 'metrics.csv' exits in log_out_folder
        metrics_file_path = os.path.join(log_out_folder, 'metrics.csv')

        if os.path.isfile(metrics_file_path):
            # Load the CSV file into a pandas DataFrame
            df = pd.read_csv(metrics_file_path, sep=',')
            # Sort the columns alphabetically
            df = df.reindex(sorted(df.columns), axis=1)
            # Define variable names
            variable_names = ['train-batch', 'train-epoch', 'train-job', 'val-batch', 'val-epoch', 'val-job']

            #Create dictionaries with filtered columns
            dicts = {}
            for var_name in variable_names:
                var_columns = [col for col in df.columns if col.startswith(var_name + '/')]
                var_dict = df[var_columns].to_dict(orient='list')
                # Remove NaN values from the dictionary
                var_dict_no_nan = {key: [value for value in values if pd.notna(value)] for key, values in var_dict.items()}
                if len(var_dict_no_nan.values())>0:
                    dicts[var_name] = var_dict_no_nan
        
            # Read 'hparams.yaml' file
            hparams_file_path = os.path.join(log_out_folder, 'hparams.yaml')
            hparams_data = {}
            if os.path.isfile(hparams_file_path):
                with open(hparams_file_path, 'r') as hparams_file:
                    hparams_data = yaml.safe_load(hparams_file)

            # Save dictionaries to Excel file with each dictionary as a sheet
            excel_file_path =  os.path.join(log_out_folder, f"job_{job_id}_report.xlsx") 

            with pd.ExcelWriter(excel_file_path, engine='xlsxwriter') as writer:
                # Save hparams to a separate sheet
                df_hparams = pd.DataFrame.from_dict(hparams_data, orient='index')
                df_hparams.to_excel(writer, sheet_name='hparams', header=False)
                
                for sheet_name, data_dict in dicts.items():
                    df_sheet = pd.DataFrame(data_dict)
                    df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)

        # Delete the 'metrics.csv' file
            #os.remove(metrics_file_path)
            print(f"Report tidied up and saved to: {excel_file_path}. 'metrics.csv' file deleted.")
        else:
            print("Error: 'metrics.csv' file not found in the specified folder.")

if __name__ == "__main__":
    pass
        
