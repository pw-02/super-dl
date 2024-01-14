import time
import threading
class Job:
    def __init__(self, job_id,dataset_id, training_speed=1):
        self.job_id = job_id
        self.job_started = time.time()
        self.batches_pending_count = 0  # Difference between batches sent and processed
        self.lock = threading.Lock()  # Lock for synchronizing access
        self.training_speed = training_speed  # Adjust processing speed as needed
        self.dataset_id = dataset_id
        self.predicted_batch_access_times = {}
        self.actial_batch_access_times = {}


    def increment_batches_pending_count(self) :
        with self.lock:
            self.batches_pending_count += 1

    def decrement_batches_prending_count(self):
        with self.lock:
            self.batches_pending_count -= 1

    def predict_batch_access_time(self, batch_id):
        with self.lock:
            current_time = time.time()
            predicted_time = current_time + (self.batches_pending_count * (1 / self.training_speed))
            self.predicted_batch_access_times[batch_id] = predicted_time
            return predicted_time
