

class TrainingJob:
    def __init__(self, job_id, job_ended=False, total_epochs=0, current_epoch_id=0, current_epoch_progress=0.0, training_speed=0.0):
        self.job_id = job_id
        self.job_ended = job_ended
        self.total_epochs = total_epochs
        self.current_epoch_id = current_epoch_id
        self.current_epoch_progress = current_epoch_progress
        self.training_speed = training_speed

    def update_status(self, job_ended=None, total_epochs=None, current_epoch_id=None, current_epoch_progress=None, training_speed=None):
        if job_ended is not None:
            self.job_ended = job_ended
        if total_epochs is not None:
            self.total_epochs = total_epochs
        if current_epoch_id is not None:
            self.current_epoch_id = current_epoch_id
        if current_epoch_progress is not None:
            self.current_epoch_progress = current_epoch_progress
        if training_speed is not None:
            self.training_speed = training_speed

    def __str__(self):
        return f"TrainingJob(job_id={self.job_id}, job_ended={self.job_ended}, total_epochs={self.total_epochs}, " \
               f"current_epoch_id={self.current_epoch_id}, current_epoch_progress={self.current_epoch_progress}, " \
               f"training_speed={self.training_speed})"

