# -*- coding: utf-8 -*-
"""
Author: Long Chen

email:lchen1307@mail.ustc.edu.cn
Date: Sun Sep  8 14:19:51 2024
University of Science and Technology of China
"""

import matplotlib.pyplot as plt

def modified_NEH(machineTimes, plot=False):
    
    """
    改进的 NEH 算法，适用于求解机台和工件数都不限的流水车间调度问题
    
    :param machineTimes: 列表中的每个元素都代表所有工件在当前机台的加工时间
    :return order: 最优序列(不能保证全局最优)
    :return makespan: 最优序列的makespan
    
    """
    
    num_jobs = len(machineTimes[0])
    num_machines = len(machineTimes)
    
    #%% Step 1: Calculate the total processing time for each job
    total_times = [sum(machineTimes[m][j] for m in range(num_machines)) for j in range(num_jobs)]
    
    #%% Step 2: Sort jobs based on the minimum total time across all machines
    # For the first machine, pick jobs with the smallest total time
    order = sorted(range(num_jobs), key=lambda x: total_times[x])
    
    #%% Step 3: Calculate the makespan using the sorted order
    start_times = [[0] * num_jobs for _ in range(num_machines)]
    completion_times = [[0] * num_jobs for _ in range(num_machines)]
    
    for i in range(num_jobs):
        job = order[i]
        for m in range(num_machines):
            if m == 0:
                # First machine: start time is the max of previous completion or current start
                start_times[m][i] = completion_times[m][i-1] if i > 0 else 0
            else:
                # Other machines: start time is the max of previous completion or current start
                start_times[m][i] = max(completion_times[m-1][i], completion_times[m][i-1] if i > 0 else 0)
            completion_times[m][i] = start_times[m][i] + machineTimes[m][job]
    
    makespan = max(completion_times[-1][-1], start_times[-1][-1])
    
    #%% Step 4: Plot the Gantt chart if plot is True
    if plot:
        fig, ax = plt.subplots(num_machines, 1, sharex=True, figsize=(10, 5))
        
        for i, job in enumerate(order):
            for m in range(num_machines):
                duration = machineTimes[m][job]
                start = start_times[m][i]
                end = start + duration
                ax[m].barh(1, duration, left=start, edgecolor='black', align='center', color=('skyblue' if m == 0 else 'lightgreen' if m == 1 else 'orange'))
                ax[m].text((start + end) / 2, 1, f'Job {job+1}', ha='center', va='center', color='black')
        
        for m in range(num_machines):
            ax[m].set_title(f'Machine {m+1}')
            ax[m].set_xlabel('Time' if m == num_machines-1 else '')
            ax[m].set_yticks([1])
            ax[m].set_yticklabels(['Machine ' + ('1' if m == 0 else '2' if m == 1 else '3')])
        
        plt.tight_layout()
        plt.show()
    
    return order, makespan

#%% Example usage:
machineTimes = [
    [18, 10, 17, 12, 16],
    [14, 19, 15, 14, 16],
    [16, 12, 11, 20, 15],
    [27, 5, 18, 20, 19]
]

order, makespan = modified_NEH(machineTimes, plot=True)
print("Optimal job order:", order)
print("Minimum makespan:", makespan)
    