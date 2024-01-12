import time
import asyncio
from superdl.logger_config import configure_logger

logger = configure_logger()

class Batch:
    def __init__(self, batch_id, batch_samples):
        self.batch_id = batch_id
        self.last_accessed = None
        self.is_cached = False
        self.samples = batch_samples
        self.next_access_time = float('inf')
        self.caching_in_progress = False
        self.lock = asyncio.Lock()

    async def set_caching_in_progress(self, in_progress: bool = False):
        async with self.lock:
            self.caching_in_progress = in_progress

    async def set_cached_status(self, is_cached: bool = False):
        async with self.lock:
            self.is_cached = is_cached
            if is_cached:
                self.last_accessed = time.time()
    
    async def update_next_access_time(self, predicted_time, next_access_duration=60 * 60):
        # Simulate an asynchronous operation (you can replace this with actual async code)
        #await asyncio.sleep(0)
        # Update next access time if the predicted time is earlier than the current next access time
        if predicted_time < self.next_access_time:
            self.next_access_time = predicted_time + next_access_duration
    

    