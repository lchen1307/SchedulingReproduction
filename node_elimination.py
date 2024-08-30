# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 10:15:07 2024

@author: Administrator
"""

def node_elimination(machine1Times, machine2Times, u, partial_schedule):
    """
    根据公式(2)来消除不满足条件的节点。
    
    :param machine1Times: 所有工件在机台1的加工时间列表
    :param machine2Times: 所有工件在机台2的加工时间列表
    :param u: 有限等待时间约束列表
    :param partial_schedule: 部分作业的调度序列
    :return: 布尔值，表示是否满足条件(2)，满足返回True，否则返回False
    """
    
    num_jobs = len(machine1Times)
    all_jobs = list(range(num_jobs))
    partial_schedule = [x - 1 for x in partial_schedule]  # 将1-based index转换为0-based index
    remaining_jobs = [job for job in all_jobs if job not in partial_schedule]
    
    # 建立remaining_jobs的原始索引和约翰逊算法输入索引的映射
    job_index_map = {remaining_jobs[i]: i for i in range(len(remaining_jobs))}
    
    # 使用约翰逊算法计算未排程工件的最优序列
    import johnson_algorithm
    
    remaining_machine1Times = [machine1Times[j] for j in remaining_jobs]
    remaining_machine2Times = [machine2Times[j] for j in remaining_jobs]
    _, _, remaining_order = johnson_algorithm.johnson_algorithm(remaining_machine1Times, remaining_machine2Times, plot=False)
    remaining_order = [x - 1 for x in remaining_order]
    
    # 将约翰逊算法的结果映射回原始索引
    reverse_job_index_map = {idx: job for job, idx in job_index_map.items()}
    original_order = [reverse_job_index_map[job] for job in remaining_order]
    
    # 将部分作业序列和未排程工件的最优序列合并
    complete_schedule_0 = partial_schedule + original_order
    complete_schedule_1 = [x + 1 for x in complete_schedule_0]
    
    # 计算每个作业在机器1和机器2上的完成时间
    import calculate_makespan
    
    _, _, completion_time_machine1, completion_time_machine2 = calculate_makespan.calculate_makespan(machine1Times, machine2Times, u, complete_schedule_1, plot=False, return_both=True)

    # 检查公式(2)的条件
    
    for i in range(len(complete_schedule_0) - 1):
        job = complete_schedule_0[i + 1]
        if not (completion_time_machine1[i] + machine1Times[job] >= completion_time_machine2[i] - u[job]):
            return False
    
    return True, complete_schedule_1


#%% 测试算例
machine1Times = [18, 10, 17, 12, 16]
machine2Times = [14, 19, 15, 14, 16]
u = [2, 4, 0, 4, 3]
partial_schedule = [2, 1]

node_elimination(machine1Times, machine2Times, u, partial_schedule)