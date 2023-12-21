from typing import Dict, List, Tuple
from urllib.parse import urlparse
from pathlib import Path
import functools
import logging
import json
import boto3

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())



class Dataset:
    def __init__(self, source_system: str, data_dir: str,):
        self.IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        self.prefix = 'train'
        self.data_dir = data_dir
        self.source_system = source_system

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
