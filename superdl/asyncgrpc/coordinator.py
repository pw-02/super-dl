import asyncio
import time
from typing import Dict
from superdl.logger_config import configure_logger
from data_objects.job import Job
from data_objects.batch import Batch
from data_objects.priority_queue import PriorityQueue
import boto3
import json
import requests

logger = configure_logger()

class Coordinator:
    def __init__(self,lambda_function_name = None, aws_region = None, testing_locally = True, s3_bucket_name = 'sdl-cifar10'):
        self.job_cache:Dict[int, Job] = {}  # Dictionary to store job information
        self.batches:Dict[int, Batch] = {}  # Dictionary to store batch information
        self.processing_event = asyncio.Event()
        self.batch_pqueue = PriorityQueue() # Priority queue for batch processing using heapq
        self.s3_bucket_name = s3_bucket_name
        self.testing_locally = testing_locally
        if lambda_function_name is not None:
            self.lambda_client = boto3.client('lambda', region_name=aws_region)
            self.lambda_function_name = lambda_function_name
        if testing_locally:
            self.sam_local_url = 'http://localhost:3000'
            self.sam_function_path = '/create_batch' 


    async def preprocess_new_batch(self, job_id, batch_id, batch_samples):
        try:
            # Check if we have seen this job before
            if job_id not in self.job_cache:
                self.job_cache[job_id] = Job(job_id)
            
            # Increment the count of batches pending for the job
            await self.job_cache[job_id].increment_sent_count()

            # Predict the time when the batch will be accessed by the job
            predicted_time = await self.job_cache[job_id].predict_batch_access_time()

            #Check if we have seen this batch_id before
            if batch_id not in self.batches:
                self.batches[batch_id] = Batch(batch_id, batch_samples)
            
            # Update next access time if the predicted time is earlier than the current next access time
            await self.batches[batch_id].update_next_access_time(predicted_time)
            
            # Enqueue a batch with its job_id and batch_id, using next_access_time as priority
            self.batch_pqueue.enqueue(self.batches[batch_id].next_access_time, (job_id, batch_id))

        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
    
    async def batch_processor(self): 
        try:
            while self.processing_event.is_set():  # Check if processing_event is set
                # Get the next batch from the priority queue
                resposne = self.batch_pqueue.dequeue()

                if resposne is None:
                    # Handle the case when the queue is empty
                    logger.info(f"Batch Processor Sleeping for {4}s - Queue Empty")
                    await asyncio.sleep(4)
                else:
                    prirotiy_value = resposne[0]
                    job_id = resposne[1][0]
                    batch_id = resposne[1][1]

                    await self.process_batch(job_id, batch_id)

                # Optionally add a delay or sleep to avoid busy-waiting
                #await asyncio.sleep(0.25)
        except Exception as e:
            logger.error(f"Error in batch_processor: {e}")
            

    async def process_batch(self, job_id, batch_id):
        try:
            batch = self.batches[batch_id]
            # Check if the batch is not already cached or os not in the process of being cached
            if not batch.is_cached and not batch.caching_in_progress:
                logger.info(f"Caching in progress for Job {job_id}, Batch {batch_id}")
                await batch.set_caching_in_progress(True)
                await self.prefetch_batch(job_id, batch_id, check_cache_first=False)
                await batch.set_caching_in_progress(False)

            # Check if it has been over 15min since the batch was last accessed
            elif (time.time() - batch.last_accessed > 900) and not batch.caching_in_progress:
                logger.info(f"Refreshing batch for Job {job_id}, Batch {batch_id}")
                await batch.set_caching_in_progress(True)
                await self.prefetch_batch(job_id, batch_id, check_cache_first=True)
                await batch.set_caching_in_progress(False)
            else:
                logger.info(f"Skipping already cached bacthed for Job {job_id}, Batch {batch_id}")

        except Exception as e:
            logger.error(f"Error in process_batch: {e}")
    

    async def prefetch_batch(self, job_id, batch_id, check_cache_first):
        try:
            logger.info(f"Creating batch and adding to cache for Job {job_id}, Batch {batch_id}")

            event_data = {
                'bucket_name': self.s3_bucket_name,
                'batch_id': batch_id,
                'batch_metadata': self.batches[batch_id].samples,  # Replace with actual batch metadata
            }

            if self.testing_locally:
                response = requests.post(f"{self.sam_local_url}{self.sam_function_path}", json=event_data)
            else:
                response = self.lambda_client.invoke(FunctionName=self.lambda_function_name,
                                                     InvocationType='Event',  # Change this based on your requirements
                                                     Payload=event_data  # Pass the required payload or input parameters
                                                     )
            # Check the response status code
            if response.status_code == 200:
                await self.batches[batch_id].set_cached_status(is_cached=True)
            else:
                print(f"Error invoking function. Status code: {response.status_code}")

        except Exception as e:
            logger.error(f"Error in prefetch_batch: {e}")

 
    async def background_task(self):
        # Background task to handle routine tasks (e.g., cache evictions)
        while True:
            await asyncio.sleep(300)  # Check every 5 minutes

            # Implement logic for cache evictions
            # ...

            # "Ping" batches to keep them alive if needed in the future
            for batch_id in self.batches:
                await self.prefetch_batch(batch_id)