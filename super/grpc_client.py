import grpc
import time
from cache_management_pb2 import TrainingMetrics
from cache_management_pb2_grpc import CacheManagementStub

class TrainingJob:
    def __init__(self, order_of_batches):
        self.order_of_batches = order_of_batches
        self.position = 0
        self.batch_count = 0
        self.last_send_time = time.time()
        self.send_interval = 60  # Set the desired interval in seconds

def send_training_metrics(training_job, training_speed, x):
    current_time = time.time()
    elapsed_time = current_time - training_job.last_send_time

    # Check if the batch count threshold or time interval has been reached
    if training_job.batch_count >= x or elapsed_time >= training_job.send_interval:
        # Determine the next x batches to send
        start_position = training_job.position
        end_position = min(start_position + x, len(training_job.order_of_batches))
        batches_to_send = training_job.order_of_batches[start_position:end_position]

        # Update the position and reset the batch count
        training_job.position = end_position
        training_job.batch_count = 0
        training_job.last_send_time = current_time

        # Update the batch count based on the number of batches sent
        training_job.batch_count += len(batches_to_send)

        # Send the batches
        channel = grpc.insecure_channel('localhost:50051')
        stub = CacheManagementStub(channel)

        metrics = TrainingMetrics(
            order_of_batches=batches_to_send,
            training_speed=training_speed
        )

        response = stub.ReportTrainingMetrics(metrics)
        print("Response:", response.message)

if __name__ == '__main__':
    # Example with periodic sending
    training_job = TrainingJob(order_of_batches=list(range(1, 100)))

    # Set the desired batch count threshold and send interval
    batch_count_threshold = 5
    send_interval = 10

    while training_job.position < len(training_job.order_of_batches):
        # Send batches when the threshold or interval is reached
        send_training_metrics(training_job, training_speed=3.0, x=batch_count_threshold)

    # Optionally, wait for any remaining batches to be sent
    time.sleep(training_job.send_interval)
    send_training_metrics(training_job, training_speed=3.0, x=batch_count_threshold)
