"""
Author: Long Chen

email:lchen1307@mail.ustc.edu.cn
Date: Sun Jul 28 13:04:57 2024
University of Science and Technology of China
"""

import heapq  # 使用heapq实现优先队列
import time
import random
node_count = 0

class Node:
    def __init__(self, job_sequence, machine1Times, machine2Times, u, lower_bound=None, makespan=None):
        self.job_sequence = job_sequence  # 当前节点的作业序列
        self.machine1Times = machine1Times
        self.machine2Times = machine2Times
        self.u = u
        self.lower_bound = lower_bound if lower_bound is not None else 0  # 节点的下界
        self.makespan = makespan if makespan is not None else float('inf')  # 节点的 makespan

    def __lt__(self, other):
        # 优先队列根据下界和 makespan 进行排序
        return (self.lower_bound, self.makespan) < (other.lower_bound, other.makespan)
    
    def calculate_makespan(self):
        _, makespan, _, _ = calculate_makespan(self.machine1Times, self.machine2Times, self.u, self.job_sequence, return_both=True)
        return makespan
    
    def calculate_lower_bound(self):
        return lower_bound(self.machine1Times, self.machine2Times, self.u, self.job_sequence)
    
    def node_elimination(self):
        return node_elimination(self.machine1Times, self.machine2Times, self.u, self.job_sequence)


#%% 1 calculate makespan

def calculate_makespan(machine1Times, machine2Times, u, order, return_both=True):
    
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
        
    if return_both:
        return makespan1, makespan2, completion_time_machine1, completion_time_machine2
    else:
        return makespan2

#%% 2 Lower Bound
def lower_bound(machine1Times, machine2Times, u, partial_schedule):
    
    """
    根据 partial schedule 计算当前调度的 lower bound
    详见论文 (CIE 1995) P66 的 Lower bound 计算公式
    
    :param partial_schedule: 已经完成排程的部分工件，但是没有排完
    
    """
    
    makespan1, makespan2, _, _ = calculate_makespan(machine1Times, machine2Times, u, partial_schedule, return_both=True)
    
    partial_schedule = [x - 1 for x in partial_schedule]
    
    remaining_machine1 = [machine1Times[i] for i in range(len(machine1Times)) if i not in partial_schedule]
    remaining_machine2 = [machine2Times[i] for i in range(len(machine2Times)) if i not in partial_schedule]
    remaining_machine1_time = sum(remaining_machine1)
    remaining_machine2_time = sum(remaining_machine2)
    
    LB1 = makespan1 + remaining_machine1_time + min(remaining_machine2)
    LB2 = max(makespan1 + min(remaining_machine1), makespan2) + remaining_machine2_time
    LB = max(LB1, LB2)
    
    return LB


#%% 3 Theorem satisfied

def is_theorem_conditions_satisfied(machine1Times, machine2Times):
    # 定理2条件：检查M2中的最大元素是否大于M1的所有元素
    theorem2_satisfied = min(machine2Times) > max(machine1Times)
    
    # 定理3条件：检查M1中的最大元素是否大于M2的所有元素
    theorem3_satisfied = min(machine1Times) > max(machine2Times)
    
    # 如果theorem2_satisfied和theorem3_satisfied都为False，则输出False
    if not theorem2_satisfied and not theorem3_satisfied:
        return False    
    return True


#%% 4 Johnson Algorithm

def johnson_algorithm(machine1Times, machine2Times):
    
    """ 
    通过约翰逊算法计算双机台流水车间问题的最优序列和makespan
    """
    
    if len(machine1Times) != len(machine2Times):
        raise ValueError("The number of machines is not match!")
    
    jobs = list(range(len(machine1Times)))
    order = [None] * len(jobs)
    
    front_index = 0
    back_index = len(jobs) - 1
    
    while jobs:
        min_job = min(jobs, key=lambda x: min(machine1Times[x], machine2Times[x]))
        
        if machine1Times[min_job] < machine2Times[min_job]:
            order[front_index] = min_job
            front_index += 1
        else:
            order[back_index] = min_job
            back_index -= 1
        
        jobs.remove(min_job)
    
    num_jobs = len(order)
    
    start_time_machine1 = [0] * num_jobs
    end_time_machine1 = [0] * num_jobs
    start_time_machine2 = [0] * num_jobs
    end_time_machine2 = [0] * num_jobs
    
    for i, job in enumerate(order):
        if i == 0:
            start_time_machine1[i] = 0
        else:
            start_time_machine1[i] = end_time_machine1[i - 1]
        end_time_machine1[i] = start_time_machine1[i] + machine1Times[job]
    
    for i, job in enumerate(order):
        if i == 0:
            start_time_machine2[i] = end_time_machine1[i]
        else:
            start_time_machine2[i] = max(end_time_machine1[i], end_time_machine2[i - 1])
        end_time_machine2[i] = start_time_machine2[i] + machine2Times[job]
    
    makespan = end_time_machine2[-1]
    
    order = [x + 1 for x in order]
    
    return order, makespan


#%% 5 Node elimination

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
    
    remaining_machine1Times = [machine1Times[j] for j in remaining_jobs]
    remaining_machine2Times = [machine2Times[j] for j in remaining_jobs]
    remaining_order, _ = johnson_algorithm(remaining_machine1Times, remaining_machine2Times)
    remaining_order = [x - 1 for x in remaining_order]
    
    # 将约翰逊算法的结果映射回原始索引
    reverse_job_index_map = {idx: job for job, idx in job_index_map.items()}
    original_order = [reverse_job_index_map[job] for job in remaining_order]
    
    # 将部分作业序列和未排程工件的最优序列合并
    complete_schedule_0 = partial_schedule + original_order
    complete_schedule_1 = [x + 1 for x in complete_schedule_0]
    
    # 计算每个作业在机器1和机器2上的完成时间
    
    _, _, completion_time_machine1, completion_time_machine2 = calculate_makespan(machine1Times, machine2Times, u, complete_schedule_1, return_both=True)

    # 检查公式(2)的条件
    
    for i in range(len(complete_schedule_0) - 1):
        job = complete_schedule_0[i + 1]
        if not (completion_time_machine1[i] + machine1Times[job] >= completion_time_machine2[i] - u[job]):
            return False, complete_schedule_1
    
    return True, complete_schedule_1


#%% 6 create child node

def create_child_nodes(node, machine1Times, machine2Times, u):
    global node_count  # 引用全局变量
    child_nodes = []
    remaining_jobs = [job for job in range(len(machine1Times)) if job not in node.job_sequence]
    for job in remaining_jobs:
        new_sequence = node.job_sequence + [job + 1]
        new_node = Node(new_sequence, machine1Times, machine2Times, u, lower_bound=node.lower_bound)
        new_node.lower_bound = new_node.calculate_lower_bound()
        new_node.makespan = new_node.calculate_makespan()
        child_nodes.append(new_node)
        node_count += 1  # 每次创建新节点时递增计数器
    return child_nodes


#%% 7 B&B main code

def branch_and_bound_algorithm(machine1Times, machine2Times, u):
    
    n = len(machine1Times)  
    # 初始化
    root = Node([], machine1Times, machine2Times, u)
    queue = [root]  # 使用列表作为优先队列
    best_solution = []
    U = float('inf')  # 最优解初始化为无穷大
    
    # 如果定理2和定理3成立，则可以直接输出最优解
    if is_theorem_conditions_satisfied(machine1Times, machine2Times):
        best_solution, U = johnson_algorithm(machine1Times, machine2Times)
    
    else:
        initial_sequence, _ = johnson_algorithm(machine1Times, machine2Times)
        U = calculate_makespan(machine1Times, machine2Times, u, initial_sequence, return_both=False)
        
        while queue and len(best_solution) < n:
            node = heapq.heappop(queue)  # 弹出下界最小的节点
            
            # 计算下界
            if node.job_sequence:
                node.lower_bound = node.calculate_lower_bound() 

            # 剪枝
            if node.lower_bound >= U:
                continue
            
            heuristic_status, complete_schedule = node.node_elimination()
            if heuristic_status:
                best_solution = complete_schedule
                U = calculate_makespan(machine1Times, machine2Times, u, complete_schedule, return_both=False)
            
            else:
                for child in create_child_nodes(node, machine1Times, machine2Times, u):
                    heapq.heappush(queue, child)
                    
                # min_makespan_node = heapq.heappop(queue) # 这句也有很大问题
                # U = min_makespan_node.lower_bound  # 这句有问题
                # best_solution = min_makespan_node.job_sequence

    return best_solution, U



#%% test data
#%%% test data 1
machine1Times = [18, 10, 17, 12, 16]
machine2Times = [14, 19, 15, 14, 16]
u = [2, 4, 0, 4, 3]

#%%% test data 2
machine1Times = [random.randint(10, 20) for _ in range(30)]
machine2Times = [random.randint(10, 20) for _ in range(30)]
u = [random.randint(0, 10) for _ in range(30)]

start_time = time.time()

best_solution, optimal_makespan = branch_and_bound_algorithm(machine1Times, machine2Times, u)

end_time = time.time()  
elapsed_time = end_time - start_time  

print(f"Elapsed time: {elapsed_time} seconds")
print(f"Optimal makespan: {optimal_makespan}")
print(f"Optimal job sequence: {best_solution}")
print(f"Total nodes generated: {node_count}")
