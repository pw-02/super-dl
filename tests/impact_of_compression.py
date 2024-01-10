import torch
import torchvision
import base64
import json
import concurrent.futures
from PIL import Image
import io
import zlib
import time
import logging
import csv
import os

logging.basicConfig(level=logging.INFO)

def download_file(bucket_name, file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
    return content

def process_file(content, transformations):
    image = Image.open(io.BytesIO(content))
    if transformations is not None:
        processed_tensor = transformations(image)
    else:
        processed_tensor = torchvision.transforms.ToTensor()(image)
    return processed_tensor

def download_and_process_file(bucket_name, file, transformations):
    file_path = file[0]
    label = file[1]
    content = download_file(bucket_name, file_path)
    processed_tensor = process_file(content, transformations)
    return processed_tensor, label

def create_torch_batch(bucket_name, batch_metadata, transformations):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(download_and_process_file, bucket_name, file, transformations): file for file in batch_metadata}
        tensor_list = []
        label_list = []
        for future in concurrent.futures.as_completed(futures):
            file_path = futures[future]
            try:
                processed_tensor, label = future.result()
                tensor_list.append(processed_tensor)
                label_list.append(label)
            except Exception as e:
                logging.error(f"Error processing file {file_path}: {e}")
    return torch.stack(tensor_list), torch.tensor(label_list)


def deserialize_transform(serialized_transform):
    transform_list = []
    for t in serialized_transform:
        transform_class = getattr(torchvision.transforms, t['type'])
        transform_instance = transform_class(**t['args'])
        transform_list.append(transform_instance)
    return torchvision.transforms.Compose(transform_list)

def encode_base64(data):
    return base64.b64encode(data).decode('utf-8')

def cal_duration(start_time, end_time, action):
    duration = end_time - start_time
    #logging.info(f"{action} Time: {duration:.5f} seconds")
    return duration

def serialize_and_deserialize_batch(event, output_file):
    bucket_name = event['bucket_name']
    file_paths = event['file_paths']

    if 'transformations' in event:
        transformations = deserialize_transform(event['transformations']['transform_params'])
    else:
        transformations = None

    tensor_batch = create_torch_batch(bucket_name, file_paths, transformations)

    buffer = io.BytesIO()
    torch.save(tensor_batch, buffer)
    serialized_data = buffer.getvalue()

    # Serialize without compression
    start_time = time.time()
    serialized_data_base64 = encode_base64(serialized_data)
    end_time = time.time()
    size_mb_base64 = len(serialized_data_base64) / (1024 ** 2)
    serialized_base64_dur =  cal_duration(start_time, end_time, "Serialization (without compression)")

    # Serialize with compression
    start_time = time.time()
    compressed_data = zlib.compress(serialized_data)
    serialized_data_base64_zlib = encode_base64(compressed_data)
    end_time = time.time()
    size_mb_base64_zlib = len(serialized_data_base64_zlib) / (1024 ** 2)
    serialized_base64_zlib_dur = cal_duration(start_time, end_time, "Serialization (with compression)")

    # Deserialize without compression
    start_time = time.time()
    deserialized_batch = dserialize_batch(serialized_data_base64, False)
    end_time = time.time()
    derialized_base64_dur = cal_duration(start_time, end_time, "Deserialization (without compression)")

    # Deserialize with compression
    start_time = time.time()
    deserialized_batch_zlib = dserialize_batch(serialized_data_base64_zlib, True)
    end_time = time.time()
    derialized_base64_zlib_dur =cal_duration(start_time, end_time, "Deserialization (with compression)")

    # Output metrics
    num_files = len(file_paths)
    metrics = {
        "num_files": num_files,
        "serialization_time_without_compression(s)": serialized_base64_dur,
        "serialization_time_with_compression(s)": serialized_base64_zlib_dur,
        "batch_size_without_compression(mb)": size_mb_base64,
        "batch_size_with_compression(mb)": size_mb_base64_zlib,
        "dserialization_time_without_compression(s)": derialized_base64_dur,
        "dserialization_time_with_compression(s)": derialized_base64_zlib_dur,
    }

    # Log metrics
    logging.info("Metrics:")
    logging.info(json.dumps(metrics, indent=2))


    with open(output_file, mode='a', newline='') as csv_file:
        fieldnames = metrics.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if csv_file.tell() == 0:
            # Write header only if file is empty
            writer.writeheader()
        writer.writerow(metrics)

    # Return metrics for additional processing
    return {
        'statusCode': 200,
        'body': metrics
    }

def dserialize_batch(serialized_data, zlib_compressed=False):
    if zlib_compressed:
        received_data_base64_zlib = base64.b64decode(serialized_data)
        decompressed_data = zlib.decompress(received_data_base64_zlib)
        buffer = io.BytesIO(decompressed_data)
        deserialized_tensor = torch.load(buffer)
    else:
        decoded_data = base64.b64decode(serialized_data)
        buffer = io.BytesIO(decoded_data)
        deserialized_tensor = torch.load(buffer)

    return deserialized_tensor

if __name__ == '__main__':
    base_path = 'datasets/vision/cifar-10/train/Airplane/aeroplane_s_000004.png',0

    test_event = {  
        'bucket_name': 'sdl-cifar10',
        # 'transformations': {
        #     'transform_params': [
        #         {'type': 'Resize', 'args': {'size': (224, 224)}},
        #         {'type': 'ToTensor', 'args': {}},
        #         # Add more transformations as needed
        #     ]
        # }
    }

  
    output_file_path = "tests/results/impact_of_compression.csv"

    # Create the directory if it doesn't exist
    output_directory = os.path.dirname(output_file_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if os.path.isfile(output_file_path):
        os.remove(output_file_path)

    batch_sizes = [1,8,16,24,32, 64, 128, 256, 512,1024, 2048]

    for batch_size in batch_sizes:
        test_event['file_paths'] = [base_path] * batch_size
        response = serialize_and_deserialize_batch(test_event, output_file_path)

