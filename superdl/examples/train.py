import grpc
import torch
from torch.utils.data import DataLoader
from your_dataset_module import YourDataset  # Replace with the actual module containing your dataset
from your_generated_proto_module import yourprotobuffile_pb2, yourprotobuffile_pb2_grpc
import custom_batch_sampler as CustomBatchSampler
# Set the fixed seed for deterministic shuffling
fixed_seed = 42

# Create the custom batch sampler with the fixed seed
custom_sampler = CustomBatchSampler(dataset_size=len(dataset), batch_size=batch_size, seed=fixed_seed)

# Create a PyTorch DataLoader with the custom sampler and multiple workers
dataloader = DataLoader(dataset, batch_sampler=custom_sampler, num_workers=4)

# gRPC service address
grpc_server_address = 'localhost:50051'

def send_batches_to_grpc(batches, batch_id):
    # Create a gRPC channel
    channel = grpc.insecure_channel(grpc_server_address)

    # Create a gRPC stub
    stub = yourprotobuffile_pb2_grpc.CacheManagementStub(channel)

    # Prepare the gRPC request
    request = yourprotobuffile_pb2.TrainingMetrics(
        order_of_batches=batches,
        training_speed=3.0,  # Replace with your actual training speed
        batch_id=batch_id
    )

    # Send the gRPC request
    response = stub.ReportTrainingMetrics(request)

    # Print the gRPC response (replace with your handling logic)
    print("gRPC Response:", response.message)

# Iterate through batches in the DataLoader
for batch, batch_id in dataloader:
    # Extract batch indices and batch ID
    batch_indices, batch_id = batch

    # Send batches to the gRPC service
    send_batches_to_grpc(batch_indices, batch_id)

    # Your PyTorch training loop logic here
