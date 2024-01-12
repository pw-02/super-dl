import grpc
import asyncio
from superdl.asyncgrpc.protos import cache_coordinator_pb2 as cache_coordinator_pb2
from superdl.asyncgrpc.protos import cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import os
import json


class SuperDLAsyncClient:
    def __init__(self):
        self.stub = None

    async def create_client(self, server_address='localhost:50051'):
        # Create a gRPC channel
        channel = grpc.aio.insecure_channel(server_address)

        # Create a gRPC stub
        self.stub = cache_coordinator_pb2_grpc.CacheCoordinatorServiceStub(channel)
    
    async def share_batch_access_pattern(self, job_id, batches):
        if self.stub is None:
            raise RuntimeError("Client not initialized. Call create_client() first.")

        # Create a BatchAccessPatternList message
        batch_access_pattern_list = cache_coordinator_pb2.BatchAccessPatternList(
            job_id=job_id,
            batches=[cache_coordinator_pb2.Batch(batch_id=batch.batch_id, batch_samples=batch.batch_samples) for batch in batches]
        )
        
        # Make the gRPC call
        response = await self.stub.ShareBatchAccessPattern(batch_access_pattern_list)

async def run_client():
    # Create a gRPC client
    client = SuperDLAsyncClient()
    await client.create_client('localhost:50051')

    job_id = os.getpid()
    single_sample = 'train/Airplane/attack_aircraft_s_001210.png', 0
    batch_size = 4
    samples = [single_sample] * batch_size
    samples = json.dumps(samples)

    batches = [
        cache_coordinator_pb2.Batch(batch_id=8, batch_samples=samples),
    ]

    # Create a list of tasks to await
    tasks = [client.share_batch_access_pattern(job_id, batches) for _ in range(5)]

    print('I am continuing with other stuff, like training a model :)')

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)

    # Continue with other tasks or exit the program
    print("Continuing with other tasks or exiting the program...")
    await asyncio.sleep(70)

if __name__ == '__main__':
    #asyncio.run(run_client())
    super_client = SuperDLAsyncClient() 
    asyncio.run(super_client.create_client())