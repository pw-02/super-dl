import time
import asyncio

class Job:
    def __init__(self, job_id, processing_speed=1):
        self.job_id = job_id
        self.job_started = time.time()
        self.batches_pending_count = 0  # Difference between batches sent and processed
        self.lock = asyncio.Lock()  # Lock for synchronizing access
        self.processing_speed = processing_speed  # Adjust processing speed as needed

    async def increment_sent_count(self):
        async with self.lock:
            self.batches_pending_count += 1

    async def increment_processed_count(self):
        async with self.lock:
            self.batches_pending_count -= 1
    
    async def predict_batch_access_time(self):
        async with self.lock:
            current_time = time.time()
            predicted_time = current_time + (self.batches_pending_count * (1 / self.processing_speed))
            return predicted_time
