import grpc
from super_dl.protos import cache_coordinator_pb2 as cache_coordinator_pb2
from super_dl.protos import cache_coordinator_pb2_grpc as cache_coordinator_pb2_grpc

class CacheCoordinatorClient:
    def __init__(self, server_address='localhost:50051'):
        # Initialize gRPC channel
        self.channel = grpc.insecure_channel(server_address)

        # Initialize CacheCoordinatorStub
        self.stub = cache_coordinator_pb2_grpc.CacheCoordinatorServiceStub(self.channel)
    
    def register_job(self,job_id,data_dir, source_system='s3'):
        job_info = cache_coordinator_pb2.JobInfo(job_id=job_id, data_dir = data_dir, source_system=source_system)
        response_register = self.stub.RegisterJob(job_info)  
        if response_register.job_is_registered:
            print(f"Registered Job with Id:'{job_id}'")
        else:
            print(f"Failed to Register Job with Id:'{job_id}'. Server Message: '{response_register.message}'.")

    def send_metrics(self, job_id, metrics_data):
        # Use self.stub to invoke RPC for sending metrics
        #metrics = MetricsData(job_id=job_id, metrics_data=metrics_data)
        #self.stub.SendMetrics(metrics)
        pass

    def send_batch_access_pattern(self,job_id, batches:list):
        batch_accesses_list = cache_coordinator_pb2.BatchAccessPatternList(job_id=job_id,
            batches=[cache_coordinator_pb2.Batch(batch_id=batch[1], batch_indices=batch[0]) for batch in batches])
        response = self.stub.SendBatchAccessPattern(batch_accesses_list)
        # Process the server response if needed
        print("Server Response:", response)

# Example usage
def main():
    server_address = "localhost:50051"
    client = CacheCoordinatorClient(server_address)

    # Use client to register a job
    job_id = client.register_job("my_training_job")
    print(f"Job registered with ID: {job_id}")

    # Use client to send metrics
    metrics_data = {"loss": 0.1, "accuracy": 95.5}
    client.send_metrics(job_id, metrics_data)
    print("Metrics sent successfully.")

    # Use client to send batch access pattern
    batch_indices = [1, 2, 3]
    client.send_batch_access_pattern(job_id, batch_indices)
    print("Batch access pattern sent successfully.")


    
if __name__ == '__main__':
 main()