import grpc
from concurrent import futures
import time
import datetime
from queue import PriorityQueue
from collections import OrderedDict
from threading import Thread
from cache_management_pb2 import TrainingMetrics, CachePreloadResponse
from cache_management_pb2_grpc import CacheManagementServicer, add_CacheManagementServicer_to_server

class Cache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.data = OrderedDict()

    def preload_batch(self, batch_id):
        if len(self.data) < self.capacity:
            self.data[batch_id] = f"Batch {batch_id} Data"

class PriorityDictQueue:
    def __init__(self):
        self.priority_queue = PriorityQueue()
        self.unique_elements = set()

    def put(self, priority, element):
        if element not in self.unique_elements:
            self.priority_queue.put((priority, element))
            self.unique_elements.add(element)

    def get(self):
        if not self.priority_queue.empty():
            priority, element = self.priority_queue.get()
            self.unique_elements.remove(element)
            return priority, element
        return None

class CacheManagementService(CacheManagementServicer):
    def __init__(self, cache, priority_dict_queue):
        self.cache = cache
        self.priority_dict_queue = priority_dict_queue

    def ReportTrainingMetrics(self, request, context):
        order_of_batches = request.order_of_batches
        training_speed = request.training_speed

        # Calculate predicted times for each batch
        predicted_times = [time.time() + i / training_speed for i in range(len(order_of_batches))]

        # Add batches and predicted times to the priority queue
        for batch_id, predicted_time in zip(order_of_batches, predicted_times):
            self.priority_dict_queue.put(predicted_time, batch_id)

        return CachePreloadResponse(message="Batches added to the processing queue")
    
    def NotifyBatchAccess(self, request, context):
        # Extract information from the gRPC request
        batch_id = request.batch_id
        cache_hit = request.cache_hit

        if batch_id in self.batch_info:
            # Update batch information
            self.batch_info[batch_id]['cached'] = cache_hit
            self.batch_info[batch_id]['last_accessed_time'] = datetime.now()

            if cache_hit:
                logging.info(f"Cache hit for Batch ID={batch_id}")
            else:
                logging.info(f"Cache miss for Batch ID={batch_id}")

            # Your additional logic based on cache hit or miss goes here

            return yourprotobuffile_pb2.Response(message=f"Notification received for Batch ID={batch_id}")
        else:
            return yourprotobuffile_pb2.Response(message=f"Batch with ID {batch_id} not found")
    
 


def pre_fetch(priority_dict_queue, cache):
    while True:
        # Retrieve the batch with the earliest predicted access time
        result = priority_dict_queue.get()
        if result is None:
            break

        _, batch_id = result

        # Simulate cache preloading
        cache.preload_batch(batch_id)

def serve():
    cache = Cache(capacity=10)
    priority_dict_queue = PriorityDictQueue()

    cache_management_service = CacheManagementService(cache, priority_dict_queue)

    # Start the pre-fetch thread
    pre_fetch_thread = Thread(target=pre_fetch, args=(priority_dict_queue, cache))
    pre_fetch_thread.daemon = True
    pre_fetch_thread.start()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_CacheManagementServicer_to_server(cache_management_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()

    try:
        while True:
            time.sleep(86400)  # Sleep for a day, as the server runs in the background
    except KeyboardInterrupt:
        server.stop(0)

