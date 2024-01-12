import grpc
import protos.cache_coordinator_pb2 as cache_coordinator_pb2 
import protos.cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import google.protobuf.empty_pb2
#from superdl.sync.coordinator import Coordinator
from asyncio import Queue
import asyncio
from superdl.logger_config import configure_logger  # Import the configured logger
from coordinator import Coordinator
import json
logger = configure_logger()  # Initialize the logger

class CacheCoordinatorService(cache_coordinator_pb2_grpc.CacheCoordinatorServiceServicer):
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator

    async def ShareBatchAccessPattern(self, request, context):
        try:
            logger.info(f"Received Batch Access Pattern for Job {request.job_id}:")
            for batch in request.batches:
                await asyncio.sleep(2)
                sampels = json.loads(batch.batch_samples)
                logger.info(f" Job {request.job_id}, Batch ID: {batch.batch_id}, Num Samples: {len(sampels)}")
                await self.coordinator.preprocess_new_batch(request.job_id, batch.batch_id,sampels)
        except Exception as e:
            logger.error(f"Error processing Batch Access Pattern: {e}")

        return google.protobuf.empty_pb2.Empty()
    
async def serve():
    coordinator = Coordinator()
    cache_service = CacheCoordinatorService(coordinator)
    coordinator.processing_event.set()
    try:
        # Start the batch processing coroutine
        asyncio.create_task(coordinator.batch_processor()) 

        # Start the background task for routine checks (e.g., cache evictions)
        #asyncio.create_task(coordinator.background_task())

        server = grpc.aio.server()
        cache_coordinator_pb2_grpc.add_CacheCoordinatorServiceServicer_to_server(cache_service, server)
        server.add_insecure_port('[::]:50051')
        await server.start()
        logger.info("Server started. Listening on port 50051...")
        
        # Keep the server running until interrupted
        await server.wait_for_termination()
    except KeyboardInterrupt:
        await server.stop(0)
    except asyncio.CancelledError:
        logger.info("Batch processing coroutine was cancelled.")
    except Exception as e:
        logger.exception(f"Error in serve(): {e}")
    finally:
        coordinator.processing_event.clear()

if __name__ == '__main__':
    # Run the server asynchronously
    asyncio.run(serve())

    # async def process_batch(self, job_id, batch):
    #     # Simulate processing time
    #     #await asyncio.sleep(7)
    #     logger.info(f"Processed Batch ID {batch.batch_id} for Job ID {job_id}")

    # async def batch_processor(self):
    #     while True:
    #         job_id, batch = await self.batch_queue.get()
    #         if job_id is None:
    #             break  # Signal to exit the coroutine
    #         await self.process_batch(job_id, batch)



