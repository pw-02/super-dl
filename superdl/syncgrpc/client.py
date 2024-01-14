import grpc
import os
import json
from superdl.syncgrpc.protos import cache_coordinator_pb2 as cache_coordinator_pb2
from superdl.syncgrpc.protos import cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
from superdl.logger_config import configure_logger

logger = configure_logger()  # Initialize the logger

class SuperClient:
    def __init__(self, server_address='localhost:50051'):
        self.stub = self.create_client(server_address)

    def create_client(self, server_address):
        # Create a gRPC channel
        channel = grpc.insecure_channel(server_address)

        # Create a gRPC stub
        return cache_coordinator_pb2_grpc.CacheCoordinatorServiceStub(channel)
    
    def register_new_job(self, job_id, dataset_id, data_source_system, data_dir):
        job_info = cache_coordinator_pb2.RegisterJobInfo(job_id=job_id, dataset_id=dataset_id,data_dir=data_dir, data_source_system=data_source_system)
        response = self.stub.RegisterJob(job_info)  
        if response.job_registered:
            logger.info(f"Registered Job with Id: '{job_id}'")
        else:
             logger.info(f"Failed to Register Job with Id: '{job_id}'. Server Message: '{response.message}'.")


    def share_batch_access_pattern(self, job_id, batches:list, batch_type):
        if self.stub is None:
            raise RuntimeError("Client not initialized. Call create_client() first.")
        batch_access_pattern_list = cache_coordinator_pb2.BatchAccessPatternList(
        job_id=job_id,
        batches=[
            cache_coordinator_pb2.Batch(batch_id=batch[1], sample_indices=batch[0])
            for batch in batches
        ],
        batch_type=batch_type)

        # Make the gRPC call
        response = self.stub.ShareBatchAccessPattern(batch_access_pattern_list)
        pass

def run_client():
    # Create a gRPC client
    client = SuperClient()
    client.create_client('localhost:50051')

    job_id = os.getpid()

       # Use client to send batch access pattern
    batch_list = [(1,[1, 2, 3]), (2,[4, 5, 6]), (3,[7, 8, 9])]
    client.share_batch_access_pattern(job_id, batch_list, 'val')

if __name__ == '__main__':
    run_client()
