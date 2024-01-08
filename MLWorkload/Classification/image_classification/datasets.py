from typing import Optional, List, Tuple, Callable, Dict
from PIL import Image
from torch.utils.data import Dataset
from .utils import is_image_file, S3Url
import torch
import functools
import base64
import gzip
import io
import json
import os
from pathlib import Path
from lightning.fabric import Fabric
import boto3

# Define constants
REDIS_PORT = 6379

class SUPERDatasetBase(Dataset):
    def __init__(self, fabric: Fabric, prefix: str, transform: Optional[Callable],cache_client,super_client ):
        self.fabric = fabric
        self.prefix = prefix
        self.cache_client = cache_client
        self.transform = transform
        self._img_classes = None
        self.super_client = super_client

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._img_classes)
            for blob in self._img_classes[blob_class]
        ]

    def __len__(self):
        return sum(len(class_items) for class_items in self._img_classes.values())

    def __getitem__(self, next_batch):
        batch_indices, batch_id = next_batch

        if self.cache_client is not None:
            cached_data = self.cache_client.get(batch_id)
        else:
            cached_data = None

        if cached_data:
            # Convert JSON batch to torch format
            torch_imgs, torch_labels = self.deserialize_torch_batch(cached_data)
            return torch_imgs, torch_labels, batch_id

        # Cache miss, load from primary storage
        images, labels = self.fetch_batch_data(batch_indices, batch_id)

        return torch.stack(images), torch.tensor(labels), batch_id

    def deserialize_torch_batch(self, batch_data):
        batch_data = base64.b64decode(batch_data)
        decompressed = gzip.decompress(batch_data)
        buffer = io.BytesIO(decompressed)
        decoded_batch = torch.load(buffer)
        batch_imgs = decoded_batch['inputs']
        batch_labels = decoded_batch['labels']
        return batch_imgs, batch_labels
    
    def convert_json_batch_to_torch_format(self, batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels = []

        for img, label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            labels.append(label)
        return torch.stack(imgs), torch.tensor(labels)
    

class SUPERLocalDataset(SUPERDatasetBase):
    def __init__(self, fabric: Fabric, prefix: str, data_dir: str, transform: Optional[Callable], cache_client, super_client):
        super().__init__(fabric, prefix, transform, cache_client, super_client)
        self._img_classes = self._classify_imgs(data_dir)
        fabric.print("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix, len(self), len(self._img_classes)))

    def _classify_imgs(self, data_dir) -> Dict[str, List[str]]:
        data_dir = str(Path(data_dir) / self.prefix)

        img_classes: Dict[str, List[str]] = {}
        index_file = Path(data_dir) / 'index.json'

        if index_file.exists():
            with open(index_file.absolute()) as f:
                img_classes = json.load(f)
        else:
            for dirpath, filenames in os.walk(data_dir):
                for filename in filter(is_image_file, filenames):
                    img_class = os.path.basename(dirpath.removesuffix('/'))
                    img_path = os.path.join(dirpath, filename)
                    img_classes.setdefault(img_class, []).append(img_path)

            json_object = json.dumps(img_classes, indent=4)
            with open(index_file, "w") as outfile:
                outfile.write(json_object)

        return img_classes
    
    def fetch_batch_data(self, batch_indices, batch_id):
        images = []
        labels = []

        for idx in batch_indices:
            path, label = self._classed_items[idx]
            img = Image.open(path)

            if img.mode == "L":
                img = img.convert("RGB")

            if self.transform is not None:
                img = self.transform(img)

            images.append(img)
            labels.append(label)

        return images, labels

class SUPERS3Dataset(SUPERDatasetBase):
    def __init__(self, fabric: Fabric, prefix: str, data_dir: str, lambda_function_name: str, transform: Optional[Callable] = None):
        super().__init__(fabric, prefix, transform)
        self._img_classes = self._classify_imgs(data_dir)
        self.lambda_function_name = lambda_function_name
        self.s3_client = boto3.client('s3')
        self.classes = self._classify_blobs_s3(S3Url(data_dir))
        fabric.print("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix, len(self), len(self._img_classes)))
    
    def fetch_batch_data(self, batch_indices, batch_id):
        self.load_from_s3(batch_id)
        s3_key = f"{self.s3_prefix}/{self.dataset_split}/{batch_id}.pt"
        s3_object = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
        s3_data = s3_object['Body'].read() 

    def _classify_blobs_s3(self, s3url: S3Url) -> Dict[str, List[str]]:
        s3_client = boto3.client('s3')
        s3_resource = boto3.resource("s3")

        try:
            # Check if 'prefix' folder exists
            response = s3_client.list_objects(Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/', MaxKeys=1)
            if 'NextMarker' not in response:
                # 'prefix' dir not found. Skipping task
                return None

            # Check if index file in the root of the folder to avoid looping through the entire bucket
            index_object = s3_resource.Object(s3url.bucket, s3url.key + 'index.json')
            try:
                file_content = index_object.get()['Body'].read().decode('utf-8')
                blob_classes = json.loads(file_content)
            except:
                # No index file found, creating it
                blob_classes = self._create_index_file_s3(s3url)

            return blob_classes
        except Exception as e:
            # Handle exceptions, e.g., log them
            print(f"Error in _classify_blobs_s3: {e}")
            return None

    def _create_index_file_s3(self, s3url: S3Url) -> Dict[str, List[str]]:
        import json

        s3_client = boto3.client('s3')
        s3_resource = boto3.resource("s3")

        blob_classes: Dict[str, List[str]] = {}
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key)

        for page in pages:
            for blob in page['Contents']:
                blob_path = blob.get('Key')
                # Check if the object is a folder which we want to ignore
                if blob_path[-1] == "/":
                    continue
                stripped_path = self._remove_prefix(blob_path, s3url.key).lstrip("/")
                # Indicates that it did not match the starting prefix
                if stripped_path == blob_path:
                    continue
                if not is_image_file(blob_path):
                    continue
                blob_class = stripped_path.split("/")[0]
                blobs_with_class = blob_classes.get(blob_class, [])
                blobs_with_class.append(blob_path)
                blob_classes[blob_class] = blobs_with_class

        index_object = s3_resource.Object(s3url.bucket, s3url.key + 'index.json')
        index_object.put(Body=(bytes(json.dumps(blob_classes, indent=4).encode('UTF-8'))))

        return blob_classes
