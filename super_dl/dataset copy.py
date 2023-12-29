import functools
import json
import logging
import redis
from typing import Dict, List, Tuple
from urllib.parse import urlparse
from pathlib import Path
import boto3
from torch import stack, tensor, save as tsave
from PIL import Image
import base64
import io
import gzip

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class Dataset:
    def __init__(self, id, source_system: str, data_dir: str,):
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        self.prefix = 'train'
        self.data_dir = data_dir
        self.source_system = source_system
        self.redis_client = redis.StrictRedis(host='localhost', port=6379, decode_responses=True)
        self.lambda_function_name = 'lambda_function_name'
        self.transform = None
        self.dataset_id = id

        if self.source_system == 's3':
            self._blob_classes = self._classify_blobs_s3(S3Url(data_dir))
        else:
            self._blob_classes = self._classify_blobs_local(data_dir)


    def __len__(self) -> int:
        return sum(len(class_items) for class_items in self._blob_classes.values())

    @functools.cached_property
    def _classed_items(self) -> List[Tuple[str, int]]:
        return [
            (blob, class_index)
            for class_index, blob_class in enumerate(self._blob_classes)
            for blob in self._blob_classes[blob_class]
        ]

    def _remove_prefix(self, s: str, prefix: str) -> str:
        if not s.startswith(prefix):
            return s
        return s[len(prefix):]

    def _classify_blobs_local(self, data_dir: str) -> Dict[str, List[str]]:
          
        blob_classes: Dict[str, List[str]] = {}

        index_file = Path(data_dir) / self.prefix / 'index.json'

        logger.info("Attempting to read data index file '{}'".format(index_file))

        if index_file.exists():
            with open(index_file, 'r') as f:
                blob_classes = json.load(f)

        total_files = sum(len(class_items) for class_items in blob_classes.values())
        logger.info(f"Finished loading {self.prefix} index. Total files: {total_files}, Total classes: {len(blob_classes)}")
        return blob_classes

    def is_image_file(self, filename: str) -> bool:
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)

    def _classify_blobs_s3(self, s3url: 'S3Url') -> Dict[str, List[str]]:
        logger.info(f"Reading index file for {s3url.url}")
        s3_client = boto3.client('s3')
        s3_resource = boto3.resource("s3")

        # check if 'prefix' folder exists
        resp = s3_client.list_objects(Bucket=s3url.bucket, Prefix=s3url.key, Delimiter='/', MaxKeys=1)
        if 'NextMarker' not in resp:
            logger.info(f"{s3url.url} dir not found. Skipping {self.prefix} task")
            return {}

        blob_classes: Dict[str, List[str]] = {}
        # check if index file in the root of the folder to avoid looping through the entire bucket
        index_file_key = s3url.key + 'index.json'
        content_object = s3_resource.Object(s3url.bucket, index_file_key)
        
        try:
            file_content = content_object.get()['Body'].read().decode('utf-8')
            blob_classes = json.loads(file_content)
        except:
            logger.info("Index file not found")

        total_files = sum(len(class_items) for class_items in blob_classes.values())
        logger.info(f"Finished loading {self.prefix} index. Total files: {total_files}, Total classes: {len(blob_classes)}")
        return blob_classes
    
    def preload_batch(self, batch_id, batch_indices):
        # Cache miss, choose between fetching from S3 or invoking Lambda function
        if self.source_system == 's3':
            images,labels = self.load_from_aws(batch_id)
        else:
            images,labels, batch_samples = self.load_from_disk(batch_indices)
            batch = self.craete_batch_for_cacehe(images,labels,)
            self.redis_client.set(batch_id, batch_samples)
            return True
    

    def craete_batch_for_cache(self, images,labels,torchFormat=True):
        
        if torchFormat:
            with io.BytesIO() as f:
                tsave({'inputs': stack(images), 'labels': tensor(labels)}, f)
                compressed = gzip.compress(f.getvalue(),compresslevel=9)
                base_64_encoded_batch = base64.b64encode(compressed).decode('utf-8')
                return base_64_encoded_batch
        else:
            samples = []
            for img, label in zip(images, labels):
                base64_encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                samples.append((base64_encoded_img, label))
    
    def load_from_disk(self, batch_indices):

        images = []
        labels = []

        for idx in batch_indices:
            path, label = self._classed_items[idx]
            file_extension = Path(path).suffix.replace('.','')      
            
            img = Image.open(path).convert("RGB")  # Convert to RGB if the image is in grayscale
            
            if self.transform is not None:
                img = self.transform(img)
            
            # Convert image to base64
            with io.BytesIO() as img_byte_arr:
                img.save(img_byte_arr, format=file_extension)
                #base64_encoded_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
                #samples.append((base64_encoded_img, label))
            
            images.append(img)
            labels.append(label)
            
            # Create payload dictionary
            #payload = {'batch_data': json.dumps(samples),'isCached': False,}
            return images, labels
        #return json.dumps(samples)


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






class S3Url:
    def __init__(self, url: str):
        self._parsed = urlparse(url, allow_fragments=False)

    @property
    def bucket(self) -> str:
        return self._parsed.netloc

    @property
    def key(self) -> str:
        if self._parsed.query:
            return self._parsed.path.lstrip('/') + '?' + self._parsed.query
        else:
            return self._parsed.path.lstrip('/')

    @property
    def url(self) -> str:
        return self._parsed.geturl()
