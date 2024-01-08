from torch import nn
from typing import ContextManager, Dict, List, Mapping, Optional, TypeVar, Union
import math
from urllib.parse import urlparse
from enum import Enum
import os
import torch.distributed
import torch.distributed as dist
from lightning.fabric import Fabric

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # noqa
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count  # noqa

    def __str__(self):
        #fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        fmtstr = "{name}:{val" + self.fmt +"}"
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
class MinMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.min = None
        self.n = 0

    def update(self, val, n=1):
        if self.min is None:
            self.min = val
        else:
            self.min = min(self.min, val)
        self.n = n

    def get_val(self):
        return self.min, self.n

    def get_data(self):
        return self.min, self.n   

class MaxMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.max = None
        self.n = 0

    def update(self, val, n=1):
        if self.max is None:
            self.max = val
        else:
            self.max = max(self.max, val)
        self.n = n

    def get_val(self):
        return self.max, self.n

    def get_data(self):
        return self.max, self.n
    

class S3Url(object):
    def __init__(self, url):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self):
        return self._parsed.netloc

    @property
    def key(self):
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self):
        return self._parsed.geturl()



def is_image_file(self, filename:str):

    IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
    any(filename.endswith(extension) for extension in IMG_EXTENSIONS)



def num_model_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    total = 0
    for p in module.parameters():
        if requires_grad is None or p.requires_grad == requires_grad:
            if hasattr(p, "quant_state"):
                # bitsandbytes 4bit layer support
                total += math.prod(p.quant_state[1])
            else:
                total += p.numel()
    return total

def get_next_exp_version(root_dir,name):
    from lightning.fabric.utilities.cloud_io import _is_dir, get_filesystem

    versions_root = os.path.join(root_dir, name)
    fs = get_filesystem(root_dir)
    if not _is_dir(fs, versions_root, strict=True):
            #log.warning("Missing logger folder: %s", versions_root)
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


def get_default_supported_precision(training: bool) -> str:
    """Return default precision that is supported by the hardware: either `bf16` or `16`.

    Args:
        training: `-mixed` or `-true` version of the precision to use

    Returns:
        default precision that is suitable for the task and is supported by the hardware
    """
    from lightning.fabric.accelerators import MPSAccelerator

    if MPSAccelerator.is_available() or (torch.cuda.is_available() and not torch.cuda.is_bf16_supported()):
        return "16-mixed" if training else "16-true"
    return "bf16-mixed" if training else "bf16-true"

def to_python_float(t:torch.Tensor)-> float:
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]
    
def reduce_tensor(tensor:torch.Tensor, fabric:Fabric):
    rt = tensor.clone().detach()
    fabric.all_reduce(rt)
    rt /= fabric.world_size()
    return rt

def calc_images_per_second(num_images, time):
    #world_size = fabric.world_size() 
    world_size = 1
    tbs = world_size * num_images
    return tbs / time

def calc_batches_per_second(num_batches, time):
    #world_size = fabric.world_size() 
    world_size = 1
    tbs = world_size * num_batches
    return tbs / time

def calc_epochs_per_second(num_epochs, time):
    #world_size = fabric.world_size() 
    world_size = 1
    tbs = world_size * num_epochs
    return tbs / time

import glob
import pandas as pd
import os

def create_job_report(job_id, folder_path):
    # Check if a file called 'summary_report.xlsx' exists in the folder
    output_file_path = os.path.join(folder_path, 'summary_report.xlsx')

    # Get a list of all CSV files in the specified folder
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

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
                if 'train/batch-id' in df_sorted.columns:
                    df_sorted['train/batch-id'] = df_sorted['train/batch-id'].astype(str).apply(lambda x: "{:.0f}".format(float(x)) if pd.notna(x) else x)
                # Convert the sorted DataFrame back to a dictionary
                sorted_data_dict = df_sorted.to_dict(orient='list')

                # Write the sorted and accumulated data for the current category to a separate sheet in the Excel file
                pd.DataFrame.from_dict(sorted_data_dict, orient='columns').to_excel(writer, sheet_name=clean_category, index=False)

if __name__ == "__main__":
    create_job_report(1, '/workspaces/super-dl/MLWorkload/Classification/logs/cifar10/version_1')
