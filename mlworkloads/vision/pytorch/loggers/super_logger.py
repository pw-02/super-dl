import csv
import logging
import os
from argparse import Namespace
from typing import Any, Dict, List, Optional, Set, Union

from torch import Tensor

from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning.fabric.utilities.logger import _add_prefix
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)

class SuperDLLogger(Logger):
    LOGGER_JOIN_CHAR = "-"

    def __init__(self, root_dir:_PATH, rank:int, prefix:str, flush_logs_every_n_steps:int = 50, name: str = "logs",version: Optional[Union[int, str]] = None):
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._name = name or ""
        self._version = version
        self._prefix = prefix
        self._fs = get_filesystem(root_dir)
        self._flush_logs_every_n_steps = flush_logs_every_n_steps
        self._rank = rank
        self._experiment: Optional[_ExperimentWriter] = None
    
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
        self._experiment = _ExperimentWriter(log_dir=self.log_dir, rank = self.rank, prefix=self._prefix)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(self, params):
        from lightning.fabric.utilities.logger import _convert_params
        params = _convert_params(params)
        self.experiment.log_hparams(params)


    #@rank_zero_only
    def log_metrics(  # type: ignore[override]
        self, metrics: Dict[str, Union[Tensor, float]], step: Optional[int] = None, metric_level = 'batch'
    ) -> None:
        prefix = f'{self._prefix}/{metric_level}'

        metrics = _add_prefix(metrics, prefix, self.LOGGER_JOIN_CHAR)
        if step is None and metric_level == 'batch':
            step = len(self.experiment.metrics)
        self.experiment.log_metrics(metrics, step)
        
        if metric_level == 'epoch' or metric_level == 'job':
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

    def __init__(self, log_dir: str, rank:int, prefix:str) -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []
        self.hparams: Dict[str, Any] = {}
        self._fs = get_filesystem(log_dir)
        self.log_dir = log_dir
        self.prefix = prefix
        if self._fs.exists(self.log_dir) and self._fs.listdir(self.log_dir):
            rank_zero_warn(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
        self._fs.makedirs(self.log_dir, exist_ok=True)
        
        self.metrics_file_path = os.path.join(self.log_dir, f'rank_{rank}_{self.prefix}_{self.NAME_METRICS_FILE}')

    def log_metrics(self, metrics_dict: Dict[str, float], step: Optional[int] = None) -> None:
        """Record metrics."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        #if step is None:
        #    step = len(self.metrics)

        metrics = {k: _handle_value(v) for k, v in metrics_dict.items()}
        if step is not None:
            metrics["step"] = step
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

