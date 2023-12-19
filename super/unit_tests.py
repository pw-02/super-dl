import unittest
from coordinator import Coordinator

class TestCoordinatorJobs(unittest.TestCase):
    def setUp(self):
        # Initialize the Coordinator before each test
        self.coordinator = Coordinator()

    def test_register_job(self):
        # Create and register a job
        self.coordinator.create_job("job1", job_ended=False, total_epochs=10)

        # Check if the job is registered
        job = self.coordinator.get_job("job1")
        self.assertIsNotNone(job, "Job not registered")

    def test_update_job_status(self):
        # Create and register a job
        self.coordinator.create_job("job1", job_ended=False, total_epochs=10)

        # Update the job status
        self.coordinator.update_job_status("job1", job_ended=True, total_epochs=15)

        # Check if the job status is updated
        job = self.coordinator.get_job("job1")
        self.assertTrue(job.job_ended, "Job status not updated")
        self.assertEqual(job.total_epochs, 15, "Total epochs not updated")

    def tearDown(self):
        # Clean up resources after each test (optional)
        pass

class TestCoordinatorDatasets(unittest.TestCase):
    def setUp(self):
        # Initialize the Coordinator before each test
        self.coordinator = Coordinator()

    def test_register_dataset(self):
        # Create and register a dataset
        self.coordinator.create_dataset("dataset1", "prefix1", "data_dir1", "s3")

        # Check if the dataset is registered
        dataset = self.coordinator.get_dataset("dataset1")
        self.assertIsNotNone(dataset, "Dataset not registered")

    def test_update_dataset_status(self):
        # Create and register a dataset
        self.coordinator.create_dataset("dataset1", "prefix1", "data_dir1", "s3")

        # Update the dataset status
        self.coordinator.update_dataset_status("dataset1", job_ended=True, total_epochs=5)

        # Check if the dataset status is updated
        dataset = self.coordinator.get_dataset("dataset1")
        self.assertTrue(dataset.job_ended, "Dataset status not updated")
        self.assertEqual(dataset.total_epochs, 5, "Total epochs not updated")

    def tearDown(self):
        # Clean up resources after each test (optional)
        pass

if __name__ == '__main__':
    unittest.main()
