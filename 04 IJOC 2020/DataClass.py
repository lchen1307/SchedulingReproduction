"""
@filename: DataClass.py
@author: Long Chen
@time: 2025-01-13
"""

class Job:
    def __init__(self, processing_time, weight, due_date):
        self.processing_time = processing_time
        self.weight = weight
        self.due_date = due_date

    def __repr__(self):

        return f"Job(processing_time={self.processing_time}, weight={self.weight}, due_date={self.due_date})"

class DataGroup:
    def __init__(self, group_id, jobs):
        self.group_id = group_id
        self.jobs = jobs

    def __repr__(self):

        return f"DataGroup(group_id={self.group_id}, jobs={self.jobs})"

class DataReader:
    @staticmethod
    def readData(path, job_count):
        data_groups = []
        group_id = 1
        job_count_per_group = job_count

        with open(path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines) and group_id <= 125:
                processing_times = []
                weights = []
                due_dates = []

                while len(processing_times) < job_count_per_group:
                    if i >= len(lines):
                        raise ValueError(f"Unexpected end of file while reading processing times for group {group_id}")

                    processing_times.extend([int(x) for x in lines[i].strip().split()])
                    i += 1

                while len(weights) < job_count_per_group:
                    if i >= len(lines):
                        raise ValueError(f"Unexpected end of file while reading weights for group {group_id}")

                    weights.extend([int(x) for x in lines[i].strip().split()])
                    i += 1

                while len(due_dates) < job_count_per_group:
                    if i >= len(lines):
                        raise ValueError(f"Unexpected end of file while reading due dates for group {group_id}")
                    due_dates.extend([int(x) for x in lines[i].strip().split()])
                    i += 1

                if len(processing_times) == job_count_per_group and len(weights) == job_count_per_group and len(due_dates) == job_count_per_group:
                    jobs = [Job(processing_times[j], weights[j], due_dates[j]) for j in range(job_count_per_group)]
                    data_groups.append(DataGroup(group_id, jobs))
                    group_id += 1

                else:
                    raise ValueError(
                        f"Data mismatch in group {group_id}. Expected {job_count_per_group} jobs, but got {len(processing_times)}, {len(weights)}, {len(due_dates)}")

        return data_groups

path = r'../Dataset/wt40.txt'
data_groups = DataReader.readData(path, job_count = 40)
for group in data_groups:
    print(group)