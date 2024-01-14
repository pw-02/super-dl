import time
from typing import Dict
from superdl.logger_config import configure_logger
from data_objects.job import Job
from data_objects.batch import Batch
from data_objects.dataset import Dataset

from data_objects.priority_queue import PriorityQueue, QueueEmpty
import threading  # Added threading module for Lock
import requests
import boto3
from concurrent import futures
#from model.real_time_model import RealTimeModel
logger = configure_logger()

class Coordinator:
    def __init__(self, lambda_function_name=None, aws_region=None, testing_locally=True, s3_bucket_name='sdl-cifar10'):
        self.job_cache: Dict[int, Job] = {}  # Dictionary to store job information
        self.datasets: Dict[int, Dataset] = {}

        self.batch_pqueue = PriorityQueue()  # Priority queue for batch processing using heapq
        self.s3_bucket_name = s3_bucket_name
        self.testing_locally = testing_locally
        #self.predict_time_model = RealTimeModel()
        if lambda_function_name is not None:
            try:
                self.lambda_client = boto3.client('lambda', region_name=aws_region)
                self.lambda_function_name = lambda_function_name
            except Exception as e:
                logger.error(f"Error initializing Lambda client: {e}")
                raise
        if testing_locally:
            self.sam_local_url = 'http://localhost:3000'
            self.sam_function_path = '/create_batch'
        
        # Initialize thread pool executors
        self.pre_process_executor = futures.ThreadPoolExecutor(max_workers=3)
        self.processing_executor =  futures.ThreadPoolExecutor(max_workers=5)
        self.post_processing_executor =  futures.ThreadPoolExecutor(max_workers=3)

        self.dequeuing_stop_event = threading.Event()
    

    def add_new_job(self,job_id, dataset_id):
        
        if job_id not in self.job_cache:
            self.job_cache[job_id] = Job(job_id, dataset_id)
            return True, "Job with Id '{}' Registered".format(job_id)
        else:
            return False, "Job with Id '{}' already exists. Not Registered.".format(job_id)
    
    def add_new_dataset(self, dataset_id, source_system, data_dir):
        try:
            dataset = Dataset(dataset_id, source_system, data_dir)
            if len(dataset) > 1:
                self.datasets[dataset_id] = dataset
                message = f"Access to dataset'{data_dir} in '{source_system}' confirmed"
                return True, message
            else:
                message = f"No data found for dataset '{data_dir}' in '{source_system}'"
                return False,  message
        except Exception as e:
            message = f"Error in preprocess_new_batch: {e}"
            logger.error(message)
            return False,  message

    def preprocess_new_batches(self, job_id, batches, batch_type):
        try:
            dataset_id = self.job_cache[job_id].dataset_id
            # Define a function to be executed asynchronously
            def prepocess_batch_async(job_id, batch_id, batch_sample_indices, dataset_id, batch_type):

                # Increment the count of batches pending for the job
                self.job_cache[job_id].increment_batches_pending_count()

                # Predict the time when the batch will be accessed by the job
                predicted_time = self.job_cache[job_id].predict_batch_access_time(batch_id)

                # Check if we have seen this batch_id before, if not add it
                dataset_batches = self.datasets[dataset_id].train_batches if batch_type == 'train' else self.datasets[dataset_id].val_batches
                if batch_id not in dataset_batches:
                    dataset_batches[batch_id] = Batch(batch_id, batch_sample_indices, batch_type)
                
                # Update next access time if the predicted time is earlier than the current next access time
                next_access_time = dataset_batches[batch_id].update_next_access_time(predicted_time)
                # Enqueue a batch with its job_id, batch_id, using next_access_time as priority
                self.batch_pqueue.enqueue(next_access_time, (job_id, dataset_id, batch_id, batch_type))

            # Submit the function for asynchronous execution
            for batch in batches:                
                self.pre_process_executor.submit(prepocess_batch_async(job_id, batch.batch_id, list(batch.sample_indices), dataset_id, batch_type ))

        except Exception as e:
            logger.error(f"Error in preprocess_new_batch: {e}")

    def batch_processor_worker(self):
        try:
            while not self.dequeuing_stop_event.is_set():
                # Dequeue items from the priority queue with a timeout
                try:
                    priority, item = self.batch_pqueue.dequeue()  # Adjust the timeout as needed
                except QueueEmpty:
                    # Handle the case when the queue is empty
                    logger.info("Batch Processor Sleeping for 3s - Queue Empty")
                    time.sleep(3)
                    continue  # Continue to the next iteration

                job_id, dataset_id, batch_id, batch_type = item
                # Use a ThreadPoolExecutor to process the item
                self.processing_executor.submit(self.process_batch, job_id, batch_id)
        except Exception as e:
            logger.error(f"Error in worker_function: {e}")
    
   
    def start_workers(self):
        self.processing_executor.submit(self.batch_processor_worker)

    def stop_workers(self):
        self.dequeuing_stop_event.set()
        self.pre_process_executor.shutdown(wait=False)
        self.processing_executor.shutdown(wait=False)
        self.post_processing_executor.shutdown(wait=False)

    def process_batch(self, job_id, batch_id, dataset_id, batch_type):
        try:
            if batch_type == 'train':
                batch = self.datasets[dataset_id].train_batches[batch_id]
            elif batch_type == 'val':
                batch = self.datasets[dataset_id].val_batches[batch_id]
            
            # Check if the batch is not already cached or is not in the process of being cached
            if not batch.is_cached and not batch.caching_in_progress:
                labelled_samples = self.datasets[dataset_id].get_samples_for_batch(batch)
                logger.info(f"Caching in progress for Job {job_id}, Batch {batch_id}")
                batch.set_caching_in_progress(True)
                self.prefetch_batch(job_id, batch_id, labelled_samples, check_cache_first=False)
                batch.set_caching_in_progress(False)

            # Check if it has been over 15min since the batch was last accessed
            elif (time.time() - batch.last_accessed > 900) and not batch.caching_in_progress:
                logger.info(f"Refreshing batch for Job {job_id}, Batch {batch_id}")
                labelled_samples = self.datasets[dataset_id].get_samples_for_batch(batch)
                batch.set_caching_in_progress(True)
                self.prefetch_batch(job_id, batch_id,labelled_samples, check_cache_first=True)
                batch.set_caching_in_progress(False)
            else:
                logger.info(f"Skipping already cached batched for Job {job_id}, Batch {batch_id}")

        except Exception as e:
            logger.error(f"Error in process_batch: {e}")

    def prefetch_batch(self, job_id, batch_id, labelled_samples, check_cache_first):
        try:
            logger.info(f"Creating batch and adding to cache for Job {job_id}, Batch {batch_id}")

            event_data = {
                'bucket_name': self.s3_bucket_name,
                'batch_id': batch_id,
                'batch_metadata': labelled_samples,  # Replace with actual batch metadata
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
                self.batches[batch_id].set_cached_status(is_cached=True)
            else:
                logger.error(f"Error invoking function. Status code: {response.status_code}")

        except Exception as e:
            logger.error(f"Error in prefetch_batch: {e}")

    def process_training_metrics(self, job_id, batch_id, metrics:Dict):
        try:
            # Define a function to be executed asynchronously
            def process_metrics_async(job_id, batch_id, metrics:Dict):
                job = self.job_cache[job_id]
                #reduce number of pending batches processed
                job.decrement_batches_prending_count()
                #update training speed
                job.training_speed = metrics['training_speed']

                #get prediction access time
                bacth_access_time = metrics['access_time']
                self.batches[batch_id].actual_access_times[job_id] = bacth_access_time
                batch_number, predicted_time = self.batches[batch_id].predicted_access_times[job_id]
                mae = abs(predicted_time - bacth_access_time)

                # Check if MAE is not None before reporting
                if mae is not None:
                    # logger.info(f"Mean Absolute Error for Sample {job_id} in Batch {batch_id}: {mae}")
                    logger.info(f"Job: {job_id}, Batch: {batch_id}: Batch Number: {batch_number}, Predited Access Time: {predicted_time},Actual Acess Time {bacth_access_time}, Mean Absolute Error: : {mae}")
                 
            # Submit the function for asynchronous execution
            self.post_processing_executor.submit(process_metrics_async)

        except Exception as e:
            logger.error(f"Error in process_training_metrics: {e}")


