# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:17:03 2024

@author: Long Chen
"""

def lower_bound(machine1Times, machine2Times, u, partial_schedule):
    
    """
    根据 partial schedule 计算当前调度的 lower bound
    详见论文 (CIE 1995) P66 的 Lower bound 计算公式
    
    :param partial_schedule: 已经完成排程的部分工件，但是没有排完
    
    """
    
    import calculate_makespan
    
    makespan1, makespan2, _, _ = calculate_makespan.calculate_makespan(machine1Times, machine2Times, u, partial_schedule, plot=False)
    
    partial_schedule = [x - 1 for x in partial_schedule]
    
    remaining_machine1 = [machine1Times[i] for i in range(len(machine1Times)) if i not in partial_schedule]
    remaining_machine2 = [machine2Times[i] for i in range(len(machine2Times)) if i not in partial_schedule]
    remaining_machine1_time = sum(remaining_machine1)
    remaining_machine2_time = sum(remaining_machine2)
    
    LB1 = makespan1 + remaining_machine1_time + min(remaining_machine2)
    LB2 = max(makespan1 + min(remaining_machine1), makespan2) + remaining_machine2_time
    LB = max(LB1, LB2)
    
    return LB

#%% 测试算例
machine1Times = [18, 10, 17, 12, 16]
machine2Times = [14, 19, 15, 14, 16]
u = [2, 4, 0, 4, 3]
partial_schedule = [5]

LB = lower_bound(machine1Times, machine2Times, u, partial_schedule)