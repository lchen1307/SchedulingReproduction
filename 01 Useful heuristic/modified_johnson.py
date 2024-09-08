# -*- coding: utf-8 -*-

"""
Created on Mon Jun  3 19:43:03 2024

@author: Long CHEN
"""

import matplotlib.pyplot as plt

def johnson_algorithm(machine1Times, machine2Times, machine3Times, plot=True):
    
    """ 
    改进的约翰逊算法，适用于求解三机台流水车间调度问题
    待调整的地方：增加序列相关设置时间，添加开关及输入参数
    
    :param plot: 是否绘制甘特图，默认绘制
    :return order: 最优序列(不能保证全局最优)
    :return makespan: 最优序列的makespan
    
    """
    
    #%% 1 将原始输入修正为两机台的形式
    machine1Times_new = [x + y for x, y in zip(machine1Times, machine2Times)]
    machine2Times_new = [x + y for x, y in zip(machine2Times, machine3Times)]
    
    #%% 2 通过Johnson Algorithm计算最优序列
    
    if len(machine1Times_new) != len(machine2Times_new):
        raise ValueError("The number of machines is not match!")
    
    jobs = list(range(len(machine1Times_new)))
    order = [None] * len(jobs)
    
    front_index = 0
    back_index = len(jobs) - 1
    
    while jobs:
        # 找到最小的元素
        min_job = min(jobs, key=lambda x: min(machine1Times_new[x], machine2Times_new[x]))
        
        if machine1Times_new[min_job] < machine2Times_new[min_job]:
            order[front_index] = min_job
            front_index += 1
        else:
            order[back_index] = min_job
            back_index -= 1
        
        jobs.remove(min_job)
    
    #%% 3 计算makespan
    
    num_jobs = len(order)
    
    start_time_machine1 = [0] * num_jobs
    end_time_machine1 = [0] * num_jobs
    start_time_machine2 = [0] * num_jobs
    end_time_machine2 = [0] * num_jobs
    start_time_machine3 = [0] * num_jobs
    end_time_machine3 = [0] * num_jobs
    
    # 计算机台1的开始和结束时间
    for i, job in enumerate(order):   # 使用enumerate来遍历order列表
        if i == 0:
            start_time_machine1[i] = 0
        else:
            start_time_machine1[i] = end_time_machine1[i - 1]
        end_time_machine1[i] = start_time_machine1[i] + machine1Times[job]
    
    # 计算机台2的开始和结束时间
    for i, job in enumerate(order):
        if i == 0:
            start_time_machine2[i] = end_time_machine1[i]
        else:
            start_time_machine2[i] = max(end_time_machine1[i], end_time_machine2[i - 1])
        end_time_machine2[i] = start_time_machine2[i] + machine2Times[job]
        
    # 计算机台3的开始和结束时间
    for i, job in enumerate(order):
        if i == 0:
            start_time_machine3[i] = end_time_machine2[i]
        else:
            start_time_machine3[i] = max(end_time_machine2[i], end_time_machine3[i - 1])
        end_time_machine3[i] = start_time_machine3[i] + machine3Times[job] 
    
    makespan = end_time_machine3[-1]
    
    #%% 4 绘制甘特图
    
    order = [x + 1 for x in order]
    
    if plot:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 5))
        
        for i, job in enumerate(order):
            ax[0].barh(1, end_time_machine1[i] - start_time_machine1[i], left=start_time_machine1[i], edgecolor='black', align='center', color='skyblue')
            ax[0].text((start_time_machine1[i] + end_time_machine1[i]) / 2, 1, f'Job {job}', ha='center', va='center', color='black')
            
            ax[1].barh(1, end_time_machine2[i] - start_time_machine2[i], left=start_time_machine2[i], edgecolor='black', align='center', color='lightgreen')
            ax[1].text((start_time_machine2[i] + end_time_machine2[i]) / 2, 1, f'Job {job}', ha='center', va='center', color='black')
            
            # 添加第三个机台的绘制逻辑
            ax[2].barh(1, end_time_machine3[i] - start_time_machine3[i], left=start_time_machine3[i], edgecolor='black', align='center', color='orange')
            ax[2].text((start_time_machine3[i] + end_time_machine3[i]) / 2, 1, f'Job {job}', ha='center', va='center', color='black')
        
        ax[0].set_title('Machine 1')
        ax[1].set_title('Machine 2')
        ax[2].set_title('Machine 3')  # 设置第三个机台的标题
        ax[2].set_xlabel('Time')  # 设置x轴标签
        
        ax[0].set_yticks([1])
        ax[1].set_yticks([1])
        ax[2].set_yticks([1])  # 设置第三个机台的y轴刻度
        
        ax[0].set_yticklabels(['Machine 1'])
        ax[1].set_yticklabels(['Machine 2'])
        ax[2].set_yticklabels(['Machine 3'])  # 设置第三个机台的y轴标签
        
        plt.tight_layout()
        plt.show()
    
    #%% 4 整理输出结果
    
    job_sequence =  ' -> '.join([f'job{x}' for x in order])
    
    return job_sequence, makespan, order

# 示例输入
machine1Times = [18, 10, 17, 12, 16]
machine2Times = [14, 19, 15, 14, 16]
machine3Times = [16, 12, 11, 20, 15]
setup_machine = 1
setup_times = [ ]

# 调用函数
optimal_order, min_makespan, order = johnson_algorithm(machine1Times, machine2Times, machine3Times)

print("最优加工次序:", optimal_order)
print("最小makespan:", min_makespan)