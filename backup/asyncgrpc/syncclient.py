import grpc
import asyncio
from superdl.asyncgrpc.protos import cache_coordinator_pb2 as cache_coordinator_pb2
from superdl.asyncgrpc.protos import cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import os
import json

async def create_client():
    # Create a gRPC channel
    channel = grpc.aio.insecure_channel('localhost:50051')  # Adjust the address based on your gRPC server configuration

    # Create a gRPC stub
    stub = cache_coordinator_pb2_grpc.CacheCoordinatorServiceStub(channel)
    return stub

async def share_batch_access_pattern(stub, job_id, batches):
    # Create a BatchAccessPatternList message
    batch_access_pattern_list = cache_coordinator_pb2.BatchAccessPatternList(
        job_id=job_id,
        batches=[cache_coordinator_pb2.Batch(batch_id=batch.batch_id, batch_samples=batch.batch_samples) for batch in batches]
    )
    # Make the gRPC call
    response = await stub.ShareBatchAccessPattern(batch_access_pattern_list)

async def main():
    # Create a gRPC client
    client_stub = await create_client()
    singple_sample = 'train/Airplane/attack_aircraft_s_001210.png',0
    batch_size  = 4
    samples = [singple_sample] * batch_size
    samples = json.dumps(samples)
    
    batches = [
        cache_coordinator_pb2.Batch(batch_id=8, batch_samples=samples),
        #cache_coordinator_pb2.Batch(batch_id=2, batch_samples=samples),
        ]

    # Create a list of tasks to await
    # Create a list of tasks to await
    tasks = [share_batch_access_pattern(client_stub, _, batches) for _ in range(5)]

    print('i am continuing with other stuff, like training a model :)')

    # Wait for all tasks to complete
    await asyncio.gather(*tasks)
    # Continue with other tasks or exit the program
    print("Continuing with other tasks or exiting the program...")


def sync_code():
    # Create an event loop
    asyncio.run(main())

# Run the synchronous code
if __name__ == '__main__':
    sync_code()
    print('moved on')