from training_job import TrainingJob
from dataset import Dataset

class Coordinator:
    def __init__(self):
        self.jobs = {}
        self.datasets = {}

    def add_job(self, job_id):
        if job_id not in self.jobs:
            self.jobs[job_id] = TrainingJob(job_id=job_id)
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
    
    def add_dataset(self, source_system, data_dir):
        sucess_response = True, "Access to dataset'{} in '{}' confirmed".format(data_dir,source_system)
        if data_dir not in self.datasets:
            dataset = Dataset(source_system, data_dir)
            if len(dataset) > 1:
                self.datasets[data_dir] = dataset
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