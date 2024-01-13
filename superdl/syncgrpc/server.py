import grpc
from concurrent import futures
import protos.cache_coordinator_pb2 as cache_coordinator_pb2
import protos.cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import google.protobuf.empty_pb2
import threading
import json
import time
from superdl.logger_config import configure_logger
from coordinator import Coordinator

logger = configure_logger()  # Initialize the logger

class CacheCoordinatorService(cache_coordinator_pb2_grpc.CacheCoordinatorServiceServicer):
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator

    def ShareBatchAccessPattern(self, request, context):
        try:
            logger.info(f"Received Batch Access Pattern for Job {request.job_id}:")
            for batch in request.batches:
                time.sleep(1)
                samples = json.loads(batch.batch_samples)
                logger.info(f" Job {request.job_id}, Batch ID: {batch.batch_id}, Num Samples: {len(samples)}")
                self.coordinator.preprocess_new_batch(request.job_id, batch.batch_id, samples)
        except Exception as e:
            logger.exception(f"Error processing Batch Access Pattern: {e}")

        return google.protobuf.empty_pb2.Empty()

def serve():
    try:
        # Create an instance of the Coordinator class
        coordinator = Coordinator()
        # Stop the batch processing  workers
        coordinator.start_workers()
        # Initialize the CacheCoordinatorService with the Coordinator instance
        cache_service = CacheCoordinatorService(coordinator)

        # Start the gRPC server
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        cache_coordinator_pb2_grpc.add_CacheCoordinatorServiceServicer_to_server(cache_service, server)
        server.add_insecure_port('[::]:50051')
        server.start()
        logger.info("Server started. Listening on port 50051...")

        # Keep the server running until interrupted
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
    except Exception as e:
        logger.exception(f"Error in serve(): {e}")
    finally:  
        # Stop the batch processing  workers
        coordinator.stop_workers()

if __name__ == '__main__':
    # Run the server
    serve()
