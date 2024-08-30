# -*- coding: utf-8 -*-
"""

Created on Sat Jul 27 09:29:42 2024
@author: Long Chen
@filename: calculate_makespan

"""

def calculate_makespan(machine1Times, machine2Times, u, order, plot=True, return_both=True):
    
    """
    根据已知排序和有限时间约束，通过递归公式(1)来计算makespan
    
    :param machine1Times: 所有工件在机台1的加工时间
    :param machine2Times: 所有工件在机台2的加工时间
    :param order: 已经计算好的加工序列
    :param u: limited waiting time constraints
    :return makespan: 考虑了时间约束后的makespan
        
    """

    order = [x - 1 for x in order]
    num_jobs = len(order) + 1
    
    # 初始化每个 job 在每台机器上的完成时间
    completion_time_machine1 = [0] * num_jobs
    completion_time_machine2 = [0] * num_jobs    
    start_time_machine1 = [0] * num_jobs
    start_time_machine2 = [0] * num_jobs
    
    # 通过(CIE 1995)的递推公式(1)来计算makespan
    for i, job in enumerate(order):
        
        # 计算在机器1上的完成时间
        completion_time_machine1[i + 1] = max(completion_time_machine1[i] + machine1Times[job],
                                              completion_time_machine2[i] - u[job]) 
        start_time_machine1[i + 1] = completion_time_machine1[i + 1] - machine1Times[job]
        
        # 计算在机器2上的完成时间
        start_time_machine2[i + 1] = max(completion_time_machine1[i + 1], completion_time_machine2[i])
        completion_time_machine2[i + 1] = start_time_machine2[i + 1] + machine2Times[job]
    
    completion_time_machine1 = completion_time_machine1[1:]
    completion_time_machine2 = completion_time_machine2[1:]    
    start_time_machine1 = start_time_machine1[1:]
    start_time_machine2 = start_time_machine2[1:]
    
    # 最后一个 job 在机器2上的完成时间就是 makespan
    makespan1 = completion_time_machine1[-1]    
    makespan2 = completion_time_machine2[-1]

    #%% 排程结果可视化
    
    import matplotlib.pyplot as plt
    
    order = [x + 1 for x in order]
    
    if plot:
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 5))
        
        for i, job in enumerate(order):
            ax[0].barh(1, completion_time_machine1[i] - start_time_machine1[i], left=start_time_machine1[i], edgecolor='black', align='center', color='skyblue')
            ax[0].text((start_time_machine1[i] + completion_time_machine1[i]) / 2, 1, f'Job {job}', ha='center', va='center', color='black')
            
            ax[1].barh(1, completion_time_machine2[i] - start_time_machine2[i], left=start_time_machine2[i], edgecolor='black', align='center', color='lightgreen')
            ax[1].text((start_time_machine2[i] + completion_time_machine2[i]) / 2, 1, f'Job {job}', ha='center', va='center', color='black')
        
        ax[0].set_title('Machine 1')
        ax[1].set_title('Machine 2')
        ax[1].set_xlabel('Time')
        ax[0].set_yticks([1])
        ax[1].set_yticks([1])
        ax[0].set_yticklabels(['Machine 1'])
        ax[1].set_yticklabels(['Machine 2'])
        
        plt.tight_layout()
        plt.show()
        
    if return_both:
        return makespan1, makespan2, completion_time_machine1, completion_time_machine2
    else:
        return makespan2

#%% 测试函数
# machine1Times = [18, 10, 17, 12, 16]
# machine2Times = [14, 19, 15, 14, 16]
# u = [2, 4, 0, 4, 3]
# order = [2, 4, 5, 3, 1]

# [makespan1, makespan2] = calculate_makespan(machine1Times, machine2Times, u, order, plot=True, return_both=True)
# print("Makespan:", makespan2)
