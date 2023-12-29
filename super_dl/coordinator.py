from training_job import TrainingJob
from dataset import Dataset
from queue import Queue, Empty
import time
import threading
import numpy as np
from concurrent.futures import ThreadPoolExecutor
#from sklearn.linear_model import LinearRegression
from super_dl.batch import Batch
from super_dl.unique_priority_queue import UniquePriorityQueue
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class Coordinator:
    def __init__(self):
        self.jobs = {}
        self.datasets: Dict[int, Dataset] = {}
        self.batches = {}
        self.batch_queue = Queue()  # Batch queue for asynchronous processing
        self.prefetch_queue = UniquePriorityQueue()  # Prefetching queue
        self.prefetch_queue_lock = threading.Lock()
        #self.prefetch_queue = {}  # Use a dictionary for prefetching queue
        self.prefetch_workers = None
        self.batch_processing_thread = None
        #self.prefetching_thread = None
        self.stop_batch_processing = threading.Event()

        #self.model = None  # Initialize the model as None --> might use later to create a model for predicing batch access time


    def add_job(self, job_id, dataset_id):
        if job_id not in self.jobs:
            self.jobs[job_id] = TrainingJob(job_id=job_id, dataset_id=dataset_id)
            return True, "Job with Id '{}' Regsistered".format(job_id)
        else:
            return False, "Job with Id '{}' already exists. Not Registered.".format(job_id)

    def get_job(self, job_id):
        return self.jobs.get(job_id)

    def update_job_status(self, job_id, **kwargs):
        job = self.jobs.get(job_id)
        if job:
            job.update_status(**kwargs)

    def get_all_jobs(self):
        return list(self.jobs.values())
    
    def get_batch(self, batch_id):
        return self.batches.get(batch_id)
    
    def add_dataset(self, dataset_id, source_system, data_dir):
        sucess_response = True, "Access to dataset'{} in '{}' confirmed".format(data_dir,source_system)
        if dataset_id not in self.datasets:
            dataset = Dataset(dataset_id, source_system, data_dir)
            if len(dataset) > 1:
                self.datasets[dataset_id] = dataset
                return sucess_response
            else:
                return False, "No data found for dataset '{}' in '{}'".format(data_dir,source_system)
        else:
            # Handle the case where the dataset already exists
                return sucess_response

    def remove_dataset(self, dataset_name):
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
    
    def get_all_datasets(self):
        return list(self.datasets.values())
    
        
    def get_dataset(self, dataset_id):
        return self.datasets[dataset_id]
    
    def start_batch_processing_thread(self):
        self.batch_processing_thread = threading.Thread(target=self.process_batches, name='batch_processing_thread')
        self.batch_processing_thread.start()
    
    def start_prefetching_workers(self, num_workers=3):
        # Use ThreadPoolExecutor for prefetching workers
        self.prefetch_workers = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='prefetch_worker')
        futures = [self.prefetch_workers.submit(self.process_prefetching) for _ in range(num_workers)]

        #self.prefetching_thread = threading.Thread(target=self.process_prefetching, name='prefetching_thread')
        #self.prefetching_thread.start()


    def stop_batch_processing_thread(self):
        self.stop_batch_processing.set()
        self.batch_processing_thread.join()
    
    def stop_prefetching_workers(self):
        if self.prefetch_workers:
            self.prefetch_workers.shutdown()
        #self.stop_batch_processing.set()
        #self.prefetching_thread.join()

    def get_or_create_batch(self, batch_id, batch_indices):  
        if batch_id not in self.batches:
            self.batches[batch_id] = Batch(batch_id=batch_id, batch_indices=batch_indices)
        return self.batches[batch_id]
    
    def process_batches(self):
        try:
            while not self.stop_batch_processing.is_set():
                try:
                    job_id, batch_info = self.batch_queue.get(timeout=1)
                    job:TrainingJob = self.get_job(job_id)
                    # job.get_batches_awaiting_processing is the number of batches into the future
                    # the target batch will be accessed and job.processing_speed is the speed
                    # at which batches are processed
                    job.increment_batches_awaiting_processing()               
                    predicted_access_time = time.time() + (job.get_batches_awaiting_processing() * (1 / job.processing_speed))

                    #logger.info(f"Predicted Access Time - Batch ID: {batch_info.batch_id},Batch Indiciess: {batch_info.batch_indices}, Predicted Time: {predicted_access_time}")
                
                    batch:Batch = self.get_or_create_batch(batch_id=batch_info.batch_id, batch_indices=batch_info.batch_indices)
                    batch.predicted_access_times[job_id] = predicted_access_time
                    if not batch.is_cached:
                        with self.prefetch_queue_lock:
                            next_access_time = batch.get_next_access_time()
                            self.prefetch_queue.push(batch.batch_id, next_access_time)

                # Perform further processing or enqueue for prefetching based on predicted time
                except Empty:
                    #time.sleep(0.5)
                    continue
        except KeyboardInterrupt:
            logger.info("Stopping batch processing thread.")
        

    def process_prefetching(self):
        try:
            while not self.stop_batch_processing.is_set():
                try:
                    batch_id = self.prefetch_queue.pop()
    
                    batch: Batch = self.get_batch(batch_id)
                    # Perform prefetching logic here
                    logger.info(f"Prefetching Batch - Batch ID: {batch.batch_id}")

                    batch_cached = first_value = next(iter(self.datasets.values()), None).preload_batch(batch.batch_id, batch.batch_indices) #for now assume there is only one dataset
                    
                    if batch_cached == True:
                        # add logic to load the batch
                        batch.set_cache_status(is_cached=True)
                        batch.set_last_pinged_time(pinged_time=time.time())

                except IndexError:
                  #time.sleep(0.5)
                  continue
                
        except KeyboardInterrupt:
            logger.info("Stopping prefetching workers.")
        except Exception as e:
            logger.error(f"Error in prefetching worker: {e}")


    '''
    def process_batches(self):
        while not self.stop_batch_processing.is_set():
            try:
                batch = self.batch_queue.get(timeout=1)
                current_system_load = 0.8  # Replace with actual system load
                job = self.jobs.get(batch.job_id)

                if self.model is None:
                    # If the model is not yet trained, train it using available data
                    data = []  # Replace with actual data collection logic
                    self.train_model(data)

                # Use the trained model to predict access time
                predicted_access_time = self.model.predict([[batch.batch_id, job.processing_speed, batch.batch_size, current_system_load]])

                print(f"Predicted Access Time - Batch ID: {batch.batch_id}, Predicted Time: {predicted_access_time}")
                # Perform further processing or enqueue for prefetching based on predicted time
            except queue.Empty:
                continue


    def train_model(self, data):
        # Process data and split into features X and labels y
        X = np.array([[entry['batch_id'], entry['processing_speed'], entry['batch_size'], entry['system_load']] for entry in data])
        y = np.array([entry['actual_access_time'] for entry in data])

        # Train the model
        self.model = LinearRegression()
        self.model.fit(X, y)

        # Save the trained model
        joblib.dump(self.model, 'trained_model.joblib')
    '''