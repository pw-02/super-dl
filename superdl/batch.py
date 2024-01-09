import threading
import time
import heapq

class Batch:
    def __init__(self, batch_id, batch_indices):
        self.batch_id: int = batch_id
        self.predicted_access_times: dict = {}
        self.actual_access_times: dict = {}
        self.is_cached: bool = False
        self.cache_satus = None
        self.batch_indices = list(batch_indices)
        self.last_pinged_time = None
        self.lock = threading.Lock()

    def get_next_access_time(self):
        sorted_items = sorted(self.predicted_access_times.items(), key=lambda x: x[1])
        if sorted_items:
            return sorted_items[0][1]
        else:
            return None  # Return None if there are no predicted access times
        
    def get_time_since_last_pinged(self):
        if self.last_pinged_time:
            return time.time() - self.last_pinged_time
        else:
            return float('inf')  # Return infinity if the batch hasn't been accessed yet

    def get_time_since_last_accssed(self):
        # Get the time since the last accessed time for a specific batch
        sorted_items = sorted(self.actual_access_times.items(), key=lambda x: x[1], reverse=True)
        if sorted_items:
            last_access_time = sorted_items[0][1]

            return time.time() - last_access_time
        else:
            return float('inf')  # Return infinity if the batch hasn't been accessed yet
    
    def set_cache_status(self, is_cached):
        with self.lock:
            self.is_cached = is_cached

    def set_last_pinged_time(self, pinged_time):
        with self.lock:
            self.last_pinged_time = pinged_time


class BatchManager:
    def __init__(self):
        self.batch_dict = {}
        self.batch_queue = []

    def get_or_create_batch(self, batch_id):
        if batch_id not in self.batch_dict:
            self.batch_dict[batch_id] = Batch(batch_id)
        return self.batch_dict[batch_id]

    def process_batch(self, job_id, batch_id, processing_speed):
        batch = self.get_or_create_batch(batch_id)
        batch.jobs_processed.add(job_id)
        batch.update_access_time(processing_speed)
        print(f"Processing Batch - Job ID: {job_id}, Batch ID: {batch_id}, "
              f"Next Access Time: {batch.next_access_time:.2f}, "
              f"Time Since Last Accessed: {batch.time_since_last_accessed:.2f}s, "
              f"Cache Status: {batch.cache_status}")

    def prefetch_batches(self, num_batches):
        # Simulate prefetching logic
        for _ in range(num_batches):
            if self.batch_queue:
                _, _, batch = heapq.heappop(self.batch_queue)
                batch.cache_status = "Prefetched"
                print(f"Prefetched Batch - Batch ID: {batch.batch_id}, "
                      f"Next Access Time: {batch.next_access_time:.2f}")
            else:
                print("No batches to prefetch.")

    def schedule_batch(self, job_id, batch_id, processing_speed, priority_factor):
        batch = self.get_or_create_batch(batch_id)
        if job_id not in batch.jobs_processed:
            heapq.heappush(self.batch_queue, (batch.next_access_time, priority_factor, batch))
        else:
            print(f"Batch - Batch ID: {batch_id} already processed by Job ID: {job_id}")

    def start_processing(self):
        while True:
            # Process batches with earliest predicted access time
            if self.batch_queue:
                _, _, batch = heapq.heappop(self.batch_queue)
                self.process_batch("Prefetching Job", batch.batch_id, 1.0)
            time.sleep(1)  # Simulate processing time

# Simulate training jobs
def simulate_training_job(batch_manager, job_id, processing_speed):
    for i in range(5):
        batch_id = f"Batch-{i}"
        priority_factor = i + 1
        batch_manager.schedule_batch(job_id, batch_id, processing_speed, priority_factor)
        time.sleep(2)  # Simulate time between batch scheduling

if __name__ == "__main__":
    # Create a BatchManager instance
    manager = BatchManager()

    # Simulate multiple training jobs with different processing speeds
    job_threads = []
    for i in range(3):
        job_id = f"Job-{i}"
        processing_speed = 2.0 / (i + 1)
        job_thread = threading.Thread(target=simulate_training_job, args=(manager, job_id, processing_speed))
        job_threads.append(job_thread)
        job_thread.start()

    # Start the processing loop for prefetching
    prefetch_thread = threading.Thread(target=manager.start_processing)
    prefetch_thread.start()

    # Wait for job threads to finish
    for job_thread in job_threads:
        job_thread.join()

    # Wait for the prefetching thread to finish
    prefetch_thread.join()
