import torch
from torch.utils.data import Dataset, DataLoader
import redis
import boto3
import json
from urllib.parse import urlparse
from typing import (Any,Callable,Optional,Dict,List,Tuple,TypeVar,Union,Iterable,)
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


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


class SUPERVisionDataset(Dataset):
    def __init__(self, source_system, cache_host, data_dir, lambda_function_name=None, prefix='train'):
        self.cache_host = cache_host
        self.prefix = prefix
        self.lambda_function_name = lambda_function_name
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG','.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',]    
        self.total_samples = 0
        # Initialize Redis client and S3 client
        self.cache_host = redis.StrictRedis(host=cache_host, port=6379, db=0)
        self.s3_client = boto3.client('s3')

        if source_system == 's3':
            self._blob_classes = self._classify_blobs_s3(S3Url(data_dir))
        else:
            self._blob_classes = self._classify_blobs_local(data_dir)
    
    def is_image_file(self, filename:str):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def _classify_blobs_local(self,data_dir) -> Dict[str, List[str]]:
        import os

        logger.info("Reading index files (all the file paths in the data_source)")
        blob_classes: Dict[str, List[str]] = {}
        
        index_file = Path(data_dir + 'index.json')
        
        if(index_file.exists()):
            f = open(index_file.absolute())
            blob_classes = json.load(f)
        else:
            logger.info("No index file found for {}, creating it..".format(data_dir))
            for dirpath, dirnames, filenames in os.walk(data_dir):
                for filename in filenames:
                    if not self.is_image_file(filename):
                        continue
                    blob_class = os.path.basename(dirpath.removesuffix('/'))
                    blobs_with_class = blob_classes.get(blob_class, [])
                    blobs_with_class.append(os.path.join(dirpath,filename))
                    blob_classes[blob_class] = blobs_with_class

            json_object = json.dumps(blob_classes, indent=4)
            with open(data_dir + 'index.json', "w") as outfile:
                outfile.write(json_object)
        self.total_samples = sum(len(class_items) for class_items in blob_classes.values())
        logger.info("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix, self.total_samples,len(blob_classes)))
        return blob_classes
    
    def _classify_blobs_s3(self,s3url:S3Url) -> Dict[str, List[str]]:
        import boto3

        s3_client = boto3.client('s3')
        s3Resource = boto3.resource("s3")
        logger.info("Reading index file (all the file paths in the data_source) for {}.".format(s3url.url))

        #check if 'prefix' folder exists
        resp = s3_client.list_objects(Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/',MaxKeys=1)
        if not 'NextMarker' in resp:
            logger.info("{} dir not found. Skipping {} task".format(s3url.url, self.prefix))
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
        
        self.total_samples = sum(len(class_items) for class_items in blob_classes.values())
        logger.info("Finished loading {} index. Total files:{}, Total classes:{}".format(self.prefix,self.total_samples,len(blob_classes)))
        return blob_classes
    

    def __len__(self):
        #return sum(len(class_items) for class_items in self._blob_classes.values())
        return self.total_samples

    def __getitem__(self, idx):
        batch_id = self.batch_order[idx]

        # Try fetching from Redis
        cached_data = self.cache_host.get(f"batch:{batch_id}")
        if cached_data:
            print(f"Cache hit for Batch ID={batch_id}")
            return torch.tensor(cached_data, dtype=torch.float32)

        # Cache miss, choose between fetching from S3 or invoking Lambda function
        if self.lambda_function_name:
            print(f"Cache miss for Batch ID={batch_id}. Invoking Lambda function...")
            self.invoke_lambda_function(batch_id)
        else:
            print(f"Cache miss for Batch ID={batch_id}. Loading from S3...")
            self.load_from_s3(batch_id)

        # Retry fetching from Redis
        cached_data = self.cache_host.get(f"batch:{batch_id}")
        if cached_data:
            print(f"Cache hit for Batch ID={batch_id} after cache miss action")
            return torch.tensor(cached_data, dtype=torch.float32)
        else:
            print(f"Cache miss action did not populate the cache for Batch ID={batch_id}")
            return None  # Handle the case when cache miss action fails to populate the cache

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

        print(f"Lambda function invoked for Batch ID={batch_id}, Response={response}")

    def load_from_s3(self, batch_id):
        s3_key = f"{self.s3_prefix}/{self.dataset_split}/{batch_id}.pt"
        s3_object = self.s3_client.get_object(Bucket=self.s3_bucket, Key=s3_key)
        s3_data = s3_object['Body'].read()
        # Store in Redis for future access
        self.cache_host.set(f"batch:{batch_id}", s3_data)

# Example usage:
batch_order = [1, 2, 3, 4, 5]  # Replace with your batch order
redis_host = 'your_redis_host'
s3_bucket = 'your_s3_bucket'
s3_prefix = 'your_s3_prefix'
lambda_function_name = 'your_lambda_function_name'  # Set to None to fetch from S3

train_dataset = SUPERVisionDataset(batch_order, redis_host, s3_bucket, s3_prefix, lambda_function_name, dataset_split='train')
val_dataset = SUPERVisionDataset(batch_order, redis_host, s3_bucket, s3_prefix, lambda_function_name, dataset_split='val')

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)

for batch in train_dataloader:
    # Your training logic goes here
    if batch is not None:
        print("Training on batch:", batch)

for batch in val_dataloader:
    # Your validation logic goes here
    if batch is not None:
        print("Validation on batch:", batch)
