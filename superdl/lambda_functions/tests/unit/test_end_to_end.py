from superdl.lambda_functions.batch_creation import app
import redis
import torch
import base64
from io import BytesIO
import zlib

def event(batch_id,base_path,batch_size):
    return {
        'bucket_name': 'sdl-cifar10',
        'batch_id': batch_id,
        'batch_metadata': [base_path] * batch_size,
        'transformations': {
            'transform_params': [
               {'type': 'Resize', 'args': {'size': (224, 224)}},
                {'type': 'ToTensor', 'args': {}},
                # Add more transformations as needed
            ]
        }
    }


def test_end_to_end_inc_cache():

    redis_client = redis.StrictRedis(host='localhost', port=6379) # Instantiate Redis client
    base_path = 'train/Airplane/attack_aircraft_s_001210.png',0
    batch_id = 1
    num_samples =  1
    new_event = event(batch_id, base_path, num_samples)
    response = app.lambda_handler(new_event, None)
    serialized_tensor_batch = redis_client.get(batch_id)
    decoded_data = base64.b64decode(serialized_tensor_batch) # Decode base64
    try:
        decoded_data = zlib.decompress(decoded_data)
    except:
        pass
      
    buffer = BytesIO(decoded_data)
    batch_samples, batch_labels = torch.load(buffer)
    pass


if __name__ == '__main__':
 test_end_to_end_inc_cache()