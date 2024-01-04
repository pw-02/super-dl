import os
import sys
import time
import logging
from urllib.parse import urlparse
from typing import Optional, Callable, Dict, List, Tuple
import json
import torch
from torch.utils.data import Dataset
import redis
import boto3
import base64
import gzip
import io
from PIL import Image
from pathlib import Path
import functools
from lightning.fabric import Fabric

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

# Define constants
REDIS_PORT = 6379
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

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


class SUPERVDataset(Dataset):
    def __init__(self, fabric: Fabric, source_system:str, cache_host:str, data_dir:str,transform:Optional[Callable] =None, lambda_function_name=None, prefix='train'):

        self.fabric = fabric
        self.prefix = prefix
        self.lambda_function_name = lambda_function_name
        # Initialize Redis client and S3 client
        if cache_host is not None:
            self.cache_host = redis.StrictRedis(host=cache_host, port=REDIS_PORT, db=0)  
            fabric.print(f"Established connection with cache. Host: {cache_host}, Port: {REDIS_PORT}")
    
        else:
            self.cache_host = None
            fabric.print(f"Not using cache")

        self.s3_client = boto3.client('s3')
        self.source_system = source_system
        self.transform = transform
        self.target_transform = None

        if source_system == 's3':
            self._blob_classes = self._classify_blobs_s3(S3Url(data_dir))
        else:
            self._blob_classes = self._classify_blobs_local(data_dir)
        
        fabric.print("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix,len(self),len(self._blob_classes)))

    
    def __len__(self):
        return sum(len(class_items) for class_items in self._blob_classes.values())
    
    def __getitem__(self, idx):
        batch_id = idx[1]
        batch_indices = idx[0]

        # Try fetching from Redis
        start_time = time.perf_counter()
        cached_data = None
        if self.cache_host is not None:
            cached_data = self.cache_host.get(batch_id)

        if cached_data:
            cache_load_time = time.perf_counter() - start_time
            decode_start_time = time.perf_counter()

            # Convert JSON batch to torch format
            #torch_imgs, torch_lables = self.convert_json_batch_to_torch_format(cached_data)
            torch_imgs, torch_lables = self.deserialize_torch_bacth(cached_data)

            decode_time = time.perf_counter() - decode_start_time

            total_load_time = time.perf_counter() - start_time
            print(f"Cache hit for Batch ID={batch_id}, Total Load Time={total_load_time}, Cache Load Time={cache_load_time}, Decode Time={decode_time}")
            return torch_imgs, torch_lables, batch_id
                
        # Cache miss, choose between fetching from S3 or invoking Lambda function
        if self.source_system == 's3':
            images,labels = self.load_from_aws(batch_id)
        else:
            #print(f"Cache miss for Batch ID={batch_id}. Loading from local disk...")
            images,labels = self.load_from_disk(batch_indices, batch_id)

        # Convert the list of images and labels to tensors
        return torch.stack(images),  torch.tensor(labels), batch_id
    
    def deserialize_torch_bacth(self,batch_data):
        batch_data = base64.b64decode(batch_data)
        decompressed = gzip.decompress(batch_data)
        buffer = io.BytesIO(decompressed)
        decoded_batch = torch.load(buffer)
        batch_imgs = decoded_batch['inputs']
        batch_labels = decoded_batch['labels']
        return batch_imgs,batch_labels
    
    def convert_json_batch_to_torch_format(self,batch_data):
        samples = json.loads(batch_data)
        imgs = []
        labels  =[]
        
        for img,label in samples:
            img = Image.open(io.BytesIO(base64.b64decode(img)))
            if self.transform is not None:
                img = self.transform(img)
            if self.target_transform is not None:
                label = self.target_transform(label)

            imgs.append(img)
            labels.append(label)

        return torch.stack(imgs), torch.tensor(labels)

    def invoke_lambda_function(self, batch_id):
        lambda_client = boto3.client('lambda')

        # Prepare payload for Lambda function
        payload = {'batch_id': batch_id}

        # Invoke Lambda function
        response = lambda_client.invoke(
            FunctionName=self.lambda_function_name,
            InvocationType='Event',  # Asynchronous invocation
            Payload=json.dumps(payload),
        )

        #print(f"Lambda function invoked for Batch ID={batch_id}, Response={response}")

    def load_from_aws(self, batch_id):
        
        if self.lambda_function_name:
            #print(f"Cache miss for Batch ID={batch_id}. Invoking Lambda function...")
            self.invoke_lambda_function(batch_id)
        else:
            #print(f"Cache miss for Batch ID={batch_id}. Loading from S3...")
            self.load_from_s3(batch_id)
            s3_key = f"{self.s3_prefix}/{self.dataset_split}/{batch_id}.pt"
            s3_object = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
            s3_data = s3_object['Body'].read()
            # Store in Redis for future access
            #self.cache_host.set(f"batch:{batch_id}", s3_data)
    
    def load_from_disk(self, batch_indices,batch_id):
        from PIL import Image
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


    def is_image_file(self, filename:str):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)
    
    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]


    def _classify_blobs_local(self,data_dir) -> Dict[str, List[str]]:
        import os

        data_dir = str(Path(data_dir) / self.prefix)

        blob_classes: Dict[str, List[str]] = {}
        index_file = Path(data_dir) / 'index.json'
        
        if(index_file.exists()):
            f = open(index_file.absolute())
            blob_classes = json.load(f)
        else:
            #logger.info("No index file found for {}, creating it..".format(data_dir))
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if not self.is_image_file(filename):
                        continue
                    blob_class = os.path.basename(dirpath.removesuffix('/'))
                    blobs_with_class = blob_classes.get(blob_class, [])
                    blobs_with_class.append(os.path.join(dirpath,filename))
                    blob_classes[blob_class] = blobs_with_class

            json_object = json.dumps(blob_classes, indent=4)
            with open(index_file, "w") as outfile:
                outfile.write(json_object)
        return blob_classes
    
    def _classify_blobs_s3(self,s3url:S3Url) -> Dict[str, List[str]]:
        import boto3

        s3_client = boto3.client('s3')
        s3Resource = boto3.resource("s3")
        logger.info("Reading index file (all the file paths in the data_source) for {}.".format(s3url.url))

        #check if 'prefix' folder exists
        resp = s3_client.list_objects(Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/',MaxKeys=1)
        if not 'NextMarker' in resp:
            #logger.info("{} dir not found. Skipping {} task".format(s3url.url, self.prefix))
            return None
        blob_classes: Dict[str, List[str]] = {}
        #check if index file in the root of the folder to avoid having to loop through the entire bucket
        content_object = s3Resource.Object(s3url.bucket, s3url.key + 'index.json')
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            blob_classes = json.loads(file_content) 
        except:
            logger.info("No index file found for {}, creating it..".format(s3url.url))
            paginator = s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=s3url.bucket, Prefix=s3url.key)        
            for page in pages:
                for blob in page['Contents']:
                    blob_path = blob.get('Key')
                    #check if the object is a folder which we want to ignore
                    if blob_path[-1] == "/":
                        continue
                    stripped_path = self._remove_prefix(blob_path, s3url.key).lstrip("/")
                    #Indicates that it did not match the starting prefix
                    if stripped_path == blob_path:
                        continue
                    if not self.is_image_file(blob_path):
                        continue
                    blob_class = stripped_path.split("/")[0]
                    blobs_with_class = blob_classes.get(blob_class, [])
                    blobs_with_class.append(blob_path)
                    blob_classes[blob_class] = blobs_with_class
                
            s3object = s3Resource.Object(s3url.bucket, s3url.key +'index.json')
            s3object.put(Body=(bytes(json.dumps(blob_classes, indent=4).encode('UTF-8'))))
            return blob_classes
    