import grpc
import protos.cache_coordinator_pb2 as cache_coordinator_pb2
import protos.cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc

def run_client():
    # Create a gRPC channel to connect to the server
    with grpc.insecure_channel('localhost:50051') as channel:
        # Create a stub (client) for the CacheCoordinatorService
        stub = cache_coordinator_pb2_grpc.CacheCoordinatorServiceStub(channel)

        # Test the RegisterJob RPC
        job_info = cache_coordinator_pb2.JobInfo(job_id=1234, data_dir = "/workspaces/super-dl/mlworkloads/vision/data/cifar-10", source_system='local')
       
        response_register = stub.RegisterJob(job_info)
        if response_register.job_is_registered:
            print(f"Registered Job with Id:'{1234}'")
        else:
            print(f"Failed to Register Job with Id:'{1234}'. Server Message: '{response_register.message}'.")

        # Test the SendMetrics RPC
        metrics_request = cache_coordinator_pb2.MetricsRequest(job_id=str(response_register.id))
        stub.SendMetrics(metrics_request)
        print("Sent metrics for the job")

        # Test the SendBatchAccessPattern RPC
        batch_access_pattern = cache_coordinator_pb2.BatchAccessPattern(
            job_id=str(response_register.id),
            batch_indices=[1, 2, 3]
        )
        stub.SendBatchAccessPattern(batch_access_pattern)
        print("Sent batch access pattern for the job")

if __name__ == '__main__':
    run_client()
