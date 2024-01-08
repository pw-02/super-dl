import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from torch import Tensor

from .utils import  AverageMeter, ProgressMeter,MinMeter, MaxMeter, calc_images_per_second,calc_batches_per_second, calc_epochs_per_second

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


class BaseMetrics:
    def __init__(self):
        self.samples_seen = AverageMeter("samples_seen", ":6.3f")
        self.data_time = AverageMeter("DataTime", ":6.3f")
        self.compute_time = AverageMeter("ComputeTime", ":6.3f")
        self.compute_ips = AverageMeter("ComputeIPS", ":6.3f")
        self.total_ips = AverageMeter("TotalIPS", ":6.3f")
        self.losses = AverageMeter("Loss", ":.4e")
        self.top1 = AverageMeter("Acc@1", ":6.2f")
        self.top5 = AverageMeter("Acc@5", ":6.2f")


class IterationMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.iteration_time = AverageMeter("TotalTime", ":6.3f")
        self.total_batches = 0
        self.iteration_seps = AverageMeter("TotalTime", ":6.3f")
    
       # progress = ProgressMeter(
       # self.total_batches, [self.iteration_time, self.data_time, self.compute_time, self.losses, self.top1, self.top5], prefix="Rank[{}]\tEpoch: [{}]".format(rank)
        #)

class EpochMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.total_batches = AverageMeter("TotalTime", ":6.3f")
        self.epoch_time = AverageMeter("TotalTime", ":6.3f")
        self.compute_bps = AverageMeter("ComputeIPS", ":6.3f")
        self.total_bps = AverageMeter("TotalIPS", ":6.3f")

class SUPERLogger(Logger):

    def __init__(self, root_dir:str, rank:int, flush_logs_every_n_steps:int, print_freq:int, exp_name: str = "logs"):
        self._root_dir = os.fspath(root_dir)
        self._name = exp_name or ""
        self._version = self._get_next_exp_version()
        self._fs = get_filesystem(root_dir)
        self._flush_logs_every_n_steps = flush_logs_every_n_steps
        self._rank = rank
        self._print_freq = print_freq
        self._experiment: Optional[_ExperimentWriter] = None
        self.iteration_metrics = IterationMetrics()
        self.epoch_metrics = {'train':EpochMetrics(), 'val':EpochMetrics()}
        self._print_freq = print_freq
        self.total_batches = 0
        self.current_epoch =0

        self.progress_meter = ProgressMeter(
        self.total_batches, [self.iteration_metrics.iteration_time, 
                        self.iteration_metrics.data_time, 
                        self.iteration_metrics.compute_time, 
                        self.iteration_metrics.losses, 
                        self.iteration_metrics.top1,
                        self.iteration_metrics.top5,
                        ], prefix="Rank[{}]\tEpoch: [{}]".format(self.rank,self.current_epoch)
    )
    
    def log_iteration_metrics(self, epoch, batch_size, step,iteration_time, data_time,
                        compute_time, compute_ips, total_ips, 
                        loss, top1, top5, batch_id, is_training:bool, iteration):
        dataset_type = 'train' if is_training else 'val'

        self.iteration_metrics.iteration_time.update(iteration_time)
        self.iteration_metrics.data_time.update(data_time)
        self.iteration_metrics.compute_time.update(compute_time)
        self.iteration_metrics.compute_ips.update(compute_ips)
        self.iteration_metrics.total_ips.update(total_ips)
        self.iteration_metrics.losses.update(loss)
        self.iteration_metrics.top1.update(top1)
        self.iteration_metrics.top5.update(top5)
        self.epoch_metrics[dataset_type].samples_seen.update(batch_size)

        self.log_metrics({"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            "batchid": str(batch_id),
                            "epoch": epoch, 
                            "batch_size": batch_size,
                            "total_time": self.iteration_metrics.iteration_time.val,
                            "data_time": self.iteration_metrics.data_time.val,
                            "compute_time": self.iteration_metrics.compute_time.val,
                            "total_ips": self.iteration_metrics.total_ips.val,
                            "compute_ips": self.iteration_metrics.compute_ips.val,
                            "loss": self.iteration_metrics.losses.val,
                            "top1": self.iteration_metrics.top1.val,
                            "top5": self.iteration_metrics.top5.val,#
                            "rank": self.rank}, 
                            step=step,
                            prefix=f'{dataset_type}.iteration',
                            force_save = False)

        if iteration is not None and (iteration + 1) % self._print_freq == 0 and self.rank == 0:
           decimal_places =4
           progress_info = {
            "epoch": f"[{epoch}]",
            "batch": f"{iteration+1}/{self.total_batches}",
            "time": round(self.iteration_metrics.iteration_time.val, decimal_places),
            "data": round(self.iteration_metrics.data_time.val, decimal_places),
            "comp": round(self.iteration_metrics.compute_time.val, decimal_places),
            "ips": round(self.iteration_metrics.total_ips.val, decimal_places),
            "cips": round(self.iteration_metrics.compute_ips.val, decimal_places),
            "loss": round(self.iteration_metrics.losses.val, decimal_places),
            "acc@1": round(self.iteration_metrics.top1.val, decimal_places)
            }
           
           self.display_progress(progress_info)


    def display_progress(self,display_info:dict[str,float]):          
        print(", ".join(f"{key}: {value}" for key, value in display_info.items()))

        # reset traing metrics

    def epoch_start(self, total_batches):
        self.total_batches = total_batches


    def epoch_end(self,epoch, is_training:bool):
        
        dataset_type = 'train' if is_training else 'val'

        self.epoch_metrics[dataset_type].epoch_time.update(self.iteration_metrics.iteration_time.sum)
        self.epoch_metrics[dataset_type].data_time.update(self.iteration_metrics.data_time.sum)
        self.epoch_metrics[dataset_type].compute_time.update(self.iteration_metrics.compute_time.sum)

        self.epoch_metrics[dataset_type].losses.update(self.iteration_metrics.losses.avg)
        self.epoch_metrics[dataset_type].top1.update(self.iteration_metrics.top1.avg)
        self.epoch_metrics[dataset_type].top5.update(self.iteration_metrics.top5.avg)
        self.epoch_metrics[dataset_type].total_batches.update(self.iteration_metrics.iteration_time.count)
        self.epoch_metrics[dataset_type].compute_ips.update(calc_images_per_second(num_images=self.epoch_metrics[dataset_type].samples_seen.val,
                                                                     time=self.epoch_metrics[dataset_type].compute_time.val))
        self.epoch_metrics[dataset_type].total_ips.update(calc_images_per_second(num_images=self.epoch_metrics[dataset_type].samples_seen.val,
                                                                     time=self.epoch_metrics[dataset_type].epoch_time.val))
        self.epoch_metrics[dataset_type].compute_bps.update(calc_batches_per_second(num_batches=self.epoch_metrics[dataset_type].total_batches.val,
                                                                     time=self.epoch_metrics[dataset_type].compute_time.val))
        self.epoch_metrics[dataset_type].total_bps.update(calc_batches_per_second(num_batches=self.epoch_metrics[dataset_type].total_batches.val,
                                                                     time=self.epoch_metrics[dataset_type].epoch_time.val))
        self.log_metrics({"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            #"idx": epoch,
                            "num_samples": self.epoch_metrics[dataset_type].samples_seen.sum,
                            "num_batches": self.epoch_metrics[dataset_type].total_batches.val,
                            "total_time": self.epoch_metrics[dataset_type].epoch_time.val,
                            "data_time": self.epoch_metrics[dataset_type].data_time.val,
                            "compute_time": self.epoch_metrics[dataset_type].compute_time.val,
                            "compute_ips": self.epoch_metrics[dataset_type].compute_ips.val,
                            "total_ips": self.epoch_metrics[dataset_type].total_ips.val,
                            "compute_bps": self.epoch_metrics[dataset_type].compute_bps.val,
                            "total_bps": self.epoch_metrics[dataset_type].total_bps.val,
                            "loss(avg)": self.epoch_metrics[dataset_type].losses.val,
                            "top1(avg)": self.epoch_metrics[dataset_type].top1.val,
                            "top5(avg)": self.epoch_metrics[dataset_type].top5.val,#
                            "rank": self.rank}, 
                            step=epoch,
                            prefix='train.epoch' if is_training else 'val.epoch',
                            force_save = True)
        if self.rank == 0:

            print_seperator_line()
            print(f"{dataset_type.upper()} EPOCH [{epoch}] SUMMARY:")
            decimal_places =3
            progress_info =  {
                    "batches": self.epoch_metrics[dataset_type].total_batches.val,
                    "time": round(self.epoch_metrics[dataset_type].epoch_time.val, decimal_places),
                    "data": round(self.epoch_metrics[dataset_type].data_time.val, decimal_places),
                    "comp": round(self.epoch_metrics[dataset_type].compute_time.val, decimal_places),
                    "tips": round(self.epoch_metrics[dataset_type].total_ips.val, decimal_places),
                    "cips": round(self.epoch_metrics[dataset_type].compute_ips.val, decimal_places),  
                    "loss": round(self.epoch_metrics[dataset_type].losses.val, decimal_places),
                    "acc@1": round(self.epoch_metrics[dataset_type].top1.val, decimal_places),
                    "acc@5": round(self.epoch_metrics[dataset_type].top5.val, decimal_places)

                }
            self.display_progress(progress_info)
            print_seperator_line()

        self.iteration_metrics = IterationMetrics()
    
    def job_end(self):

        for dataset_type in self.epoch_metrics.keys():
            if self.epoch_metrics[dataset_type].compute_bps.count > 0:
                self.log_metrics({"timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            "num_samples": self.epoch_metrics[dataset_type].samples_seen.sum,
                            "num_batches": self.epoch_metrics[dataset_type].samples_seen.count,
                            "total_time": self.epoch_metrics[dataset_type].epoch_time.sum,
                            "data_time": self.epoch_metrics[dataset_type].data_time.sum,
                            "compute_time": self.epoch_metrics[dataset_type].compute_time.sum,
                            "compute_ips": calc_images_per_second(num_images=self.epoch_metrics[dataset_type].samples_seen.sum,
                                                                     time=self.epoch_metrics[dataset_type].compute_time.sum),
                            "total_ips": calc_images_per_second(num_images=self.epoch_metrics[dataset_type].samples_seen.sum,
                                                                     time=self.epoch_metrics[dataset_type].epoch_time.sum),
                            "compute_bps": calc_batches_per_second(num_batches=self.epoch_metrics[dataset_type].total_batches.sum,
                                                                     time=self.epoch_metrics[dataset_type].compute_time.sum),
                            "total_bps":calc_batches_per_second(num_batches=self.epoch_metrics[dataset_type].total_batches.sum,
                                                                     time=self.epoch_metrics[dataset_type].epoch_time.sum),
                            "compute_eps": calc_epochs_per_second(num_epochs=self.epoch_metrics[dataset_type].samples_seen.count,
                                                                     time=self.epoch_metrics[dataset_type].compute_time.sum),
                            "total_eps":calc_epochs_per_second(num_epochs=self.epoch_metrics[dataset_type].samples_seen.sum,
                                                                     time=self.epoch_metrics[dataset_type].epoch_time.sum),
                            "loss(avg)": self.epoch_metrics[dataset_type].losses.avg,
                            "top1(avg)": self.epoch_metrics[dataset_type].top1.avg,
                            "top5(avg)": self.epoch_metrics[dataset_type].top5.avg,#
                            "rank": self.rank}, 
                            step=1,
                            prefix='train.job' if dataset_type == 'train' else 'val.job',
                            force_save = True)
    
    @rank_zero_only
    def create_job_report(self):
        import glob
        import pandas as pd
        import os
        # Check if a file called 'summary_report.xlsx' exists in the folder
        output_file_path = os.path.join(self.log_dir, 'summary_report.xlsx')

        # Get a list of all CSV files in the specified folder
        csv_files = glob.glob(os.path.join(self.log_dir, '*.csv'))

        # Specify column categories
        categories = ['train.iteration', 'train.epoch', 'train.job', 'val.iteration', 'val.epoch', 'val.job']

        # Create a dictionary to store accumulated data for each category
        category_data = {category: {} for category in categories}

        # Iterate through each CSV file
        for csv_file in csv_files:
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file)

            # Iterate through each category
            for category in categories:
                # Select columns starting with the current category
                selected_columns = [col for col in df.columns if col.startswith(category)]

                # Update the dictionary for the current category with the selected columns
                if selected_columns:

                    data_dict = df[selected_columns].to_dict(orient='list')

                    # Remove NaN values from the dictionary
                    data_dict_no_nan = {key: [value for value in values if pd.notna(value)] for key, values in data_dict.items()}

                    # Accumulate data for the current category across all CSV files
                    if category_data[category]:
                        for key, values in data_dict_no_nan.items():
                            category_data[category][key].extend(values)
                    else:
                        category_data[category] = data_dict_no_nan

        # Create an Excel writer for the output file
        with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
            # Iterate through each category
            for category, data_dict in category_data.items():
                # Skip creating sheets if the data for the current category is empty
                if data_dict:
                    # Replace invalid characters in the sheet name
                    clean_category = category.replace('/', '_').replace('\\', '_').replace('*', '').replace('?', '').replace(':', '').replace('[', '').replace(']', '')

                    # Convert the dictionary to a DataFrame
                    df_sorted = pd.DataFrame.from_dict(data_dict, orient='columns')

                    # Sort the DataFrame by the 'timestamp' column
                    timestamp_column = next((col for col in data_dict.keys() if 'timestamp' in col.lower()), None)
                    if timestamp_column:
                        df_sorted.sort_values(by=timestamp_column, inplace=True)
                                    # Change the data type of 'train/batch-id' column to string
                    # if 'train/batch-id' in df_sorted.columns:
                    #     df_sorted['train/batch-id'] = df_sorted['train/batch-id'].astype(str).apply(lambda x: "{:.0f}".format(float(x)) if pd.notna(x) else x)
                    # # Convert the sorted DataFrame back to a dictionary
                    sorted_data_dict = df_sorted.to_dict(orient='list')

                    # Write the sorted and accumulated data for the current category to a separate sheet in the Excel file
                    pd.DataFrame.from_dict(sorted_data_dict, orient='columns').to_excel(writer, sheet_name=clean_category, index=False)


    def _get_next_exp_version(self):
        from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

        versions_root = os.path.join(self.root_dir, self.name)
        fs = get_filesystem(self.root_dir)
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



    @property
    def rank(self) -> str:
        """Gets the rank of the experiment.
        Returns:
            The rank of the experiment.
        """
        return self._rank


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
        """Gets the save directory where the versioned experiments are saved."""
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
    
    @property
    #@rank_zero_experiment
    def experiment(self) -> "_ExperimentWriter":
        """Actual ExperimentWriter object. To use ExperimentWriter features anywhere in your code, do the following.

        Example::

            self.logger.experiment.some_experiment_writer_function()

        """
        if self._experiment is not None:
            return self._experiment

        os.makedirs(self._root_dir, exist_ok=True)
        self._experiment = _ExperimentWriter(log_dir=self.log_dir, rank = self.rank)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        from lightning.fabric.utilities.logger import _convert_params
        params = _convert_params(params)
        self.experiment.log_hparams(params)


    #@rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Union[Tensor, float]],prefix:str, step:int, force_save:bool = False
    ) -> None:
        if step is not None:
            metrics["step"] = step
        metrics = _add_prefix(metrics, prefix, '.')
        
        self.experiment.log_metrics(metrics)

        if force_save:
            self.save()
        elif step is not None and (step + 1) % self._flush_logs_every_n_steps == 0:
            self.save()

    #@rank_zero_only
    def save(self)-> None: 
        super().save()
        self.experiment.save()

    #@rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is None:
            # When using multiprocessing, finalize() should be a no-op on the main process, as no experiment has been
            # initialized there
            return
        self.save()
    
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


class _ExperimentWriter:
    r"""Experiment writer for CSVLogger.

    Args:
        log_dir: Directory for the experiment logs

    """
    NAME_METRICS_FILE = "metrics.csv"
    NAME_HPARAMS_FILE = "hparams.yaml"

    def __init__(self, log_dir: str, rank:int) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []
        self.hparams: Dict[str, Any] = {}
        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        self._fs.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_file_path = os.path.join(self.log_dir, f'rank_{rank}_{self.NAME_METRICS_FILE}')

    def log_metrics(self, metrics_dict: Dict[str, float]) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = self._fs.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with self._fs.open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with self._fs.open(self.metrics_file_path, "r", newline="") as file:
            metrics = list(csv.DictReader(file))

        with self._fs.open(self.metrics_file_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)
    
    def log_hparams(self, params: Dict[str, Any]) -> None:
        from lightning.pytorch.core.saving import save_hparams_to_yaml
        #"""Record hparams."""
        #self.hparams.update(params)
        """Save recorded hparams and metrics into files."""
        hparams_file = os.path.join(self.log_dir, self.NAME_HPARAMS_FILE)
        save_hparams_to_yaml(hparams_file, params)

def print_seperator_line():
            print('-' * 100)