import grpc
from concurrent import futures
import protos.cache_coordinator_pb2 as cache_coordinator_pb2 
import protos.cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import google.protobuf.empty_pb2
from coordinator import Coordinator
import hashlib

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

class CacheCoordinatorService(cache_coordinator_pb2_grpc.CacheCoordinatorServiceServicer):
    def __init__(self, coordinator):
        self.coordinator:Coordinator = coordinator

    def RegisterJob(self, request, context):
        #First confirm access to the data source
        combined_str = f"{request.source_system}_{request.data_dir}"
        dataset_id = hashlib.sha256(combined_str.encode()).hexdigest()
        
        data_accessible, message = self.coordinator.add_dataset(dataset_id = dataset_id, source_system= request.source_system, data_dir= request.data_dir)
        if not data_accessible:
            return cache_coordinator_pb2.RegisterJobResponse(message = message, job_is_registered = False)    
        #now register the job, it will not register if the job already exists
        job_registered, message = self.coordinator.add_job(request.job_id, dataset_id=dataset_id)
        return cache_coordinator_pb2.RegisterJobResponse(message = message, job_is_registered = job_registered)
    
    def SendBatchAccessPattern(self, request, context):
        # Implement SendBatchAccessPattern logic
        logger.info(f"Received Batch Access Pattern for Job {request.job_id}:")
        for batch in request.batches:
            self.coordinator.batch_queue.put((request.job_id,batch))
            #logger.info(f"Batch ID: {batch.batch_id}, Indices: {batch.batch_indices}")
        return google.protobuf.empty_pb2.Empty()
    

    def SendMetrics(self, request, context):
        # Implement SendMetrics logic
        return google.protobuf.empty_pb2.Empty()


def serve():
    
    # Create an instance of the Coordinator class
    coordinator = Coordinator()

    # Initialize the CacheCoordinatorService with the Coordinator instance
    cache_service = CacheCoordinatorService(coordinator)
   
    # Start the batch processing thread
    coordinator.start_batch_processing_thread()

    # Start the prefetching workers
    coordinator.start_prefetching_workers(num_workers=8)



    # Start the gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cache_coordinator_pb2_grpc.add_CacheCoordinatorServiceServicer_to_server(cache_service, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started. Listening on port 50051...")
    server.wait_for_termination()

    # Stop the batch processing thread when the server exits
    coordinator.stop_batch_processing_thread()
    
    # Stop the prefetching workers
    coordinator.stop_prefetching_workers()

if __name__ == '__main__':
    serve()
