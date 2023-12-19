import grpc
from concurrent import futures
import protos.cache_coordinator_pb2 as cache_coordinator_pb2 
import protos.cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc
import google.protobuf.empty_pb2
from coordinator import Coordinator

class CacheCoordinatorService(cache_coordinator_pb2_grpc.CacheCoordinatorServiceServicer):
    def __init__(self):
        self.coordinator = Coordinator()

    def RegisterJob(self, request, context):
        #First confirm access to the data source
        data_accessible, message = self.coordinator.add_dataset(request.source_system, request.data_dir)
        if not data_accessible:
            return cache_coordinator_pb2.RegisterJobResponse(message = message, job_is_registered = False)    
        #now register the job, it will not register if the job already exists
        job_registered, message = self.coordinator.add_job(request.job_id)
        return cache_coordinator_pb2.RegisterJobResponse(message = message, job_is_registered = job_registered)
    
    def SendBatchAccessPattern(self, request, context):
        # Implement SendBatchAccessPattern logic
        print(f"Received BatchAccessPattern for Job {request.job_id}: {request.batch_indices}")
        return google.protobuf.empty_pb2.Empty()
    
    




    def SendMetrics(self, request, context):
        # Implement SendMetrics logic
        return google.protobuf.empty_pb2.Empty()



def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    cache_coordinator_pb2_grpc.add_CacheCoordinatorServiceServicer_to_server(CacheCoordinatorService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("Server started. Listening on port 50051...")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
