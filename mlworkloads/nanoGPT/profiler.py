import os
from datetime import datetime
class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0  # noqa
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n

        self.count += n
        self.avg = self.sum / self.count  # noqa

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_to_file = True, log_dir  = "", log_name =""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        date_raw = datetime.now()
        date_processed = "{}-{}-{}_{}-{}-{}".format(date_raw.year, date_raw.month, date_raw.day, date_raw.hour, date_raw.minute, date_raw.second)
        #self.log_file_path = 'mlworkloads/nanoGPT/logs/nano_gpt_' + date_processed + '.csv'
        self.log_file_path = log_dir + "/" + log_name + '_' + date_processed + '.csv'
        self.log_to_file = log_to_file
        self.round_to = 3

        if self.log_to_file:
            header_line = "iter,"
            for meter in self.meters:
                header_line += meter.name + '(val),'
            for meter in self.meters:
                header_line += meter.name + '(avg),'
        if not os.path.isfile(self.log_file_path):
            logfile = open(self.log_file_path, 'w')
            logfile.write(header_line.rstrip(",") + "\n")
            logfile.close()
           

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        if self.log_to_file:
            self.write_to_file(batch)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"
    
    def write_to_file(self, batch):
        line = str(batch) + ","

        for meter in self.meters:
            line += str(round(meter.val,self.round_to)) + ','
        for meter in self.meters:
            line += str(round(meter.avg, self.round_to)) + ','
        file1 = open(self.log_file_path, "a")  # append mode
        file1.write(line.rstrip(",") + "\n")
        file1.close()
        #assert()
    
    def reset(self):
        for meter in self.meters:
            meter.reset()