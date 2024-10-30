"""

@filename: ATIF_Solver.py
@author: Long Chen
@time: 2025-01-09

"""

from pyscipopt import Model, quicksum
import numpy as np
import matplotlib.pyplot as plt

def ATIF_Solving(processing_times, weights, num_machines, relaxation = False):

    """
    调用求解器求解平行机调度问题的 arc-time-indexed formulation

    主要参考论文：
    (JOC 2020) An Improved Branch-Cut-and-Price Algorithm for Parallel Machine Scheduling Problems

    :param processing_times: list 加工时间
    :param weight: list 每个工件进行加工的权重
    :param num_machines: 机器的数量
    :param relaxation: 求解线性松弛还是直接求到整数解
    :param plot: 是否将函数求解结果可视化

    :return solution, schedule, optimal_alpha

    """

    num_jobs = len(processing_times)
    processing_times = [0] + processing_times
    weights = [0] + weights
    T = sum(processing_times)
    model = Model('ATIF_Scheduling')

    # 决策变量
    x = {}

    for i in range(num_jobs + 1):
        for j in range(num_jobs + 1):
            if (i == 0) & (j == 0):
                for t in range(0, T + 1):
                    x[i, j, t] = model.addVar(vtype = 'C' if (relaxation==True) else 'I', name = f'x_{i}_{j}_{t}')

            elif i != j:
                for t in range(processing_times[i], T - processing_times[j] + 1):
                    x[i, j, t] = model.addVar(vtype = 'C' if (relaxation==True) else 'I', name = f'x_{i}_{j}_{t}')

    # 如下决策变量是必要的，但是原论文可能忘记定义了
    for j in range(num_jobs + 1):
        x[0, j, 0] = model.addVar(vtype = 'C' if (relaxation==True) else 'I', name = f'x_{0}_{j}_{0}')

    # 目标函数
    objective = 0
    for i in range(num_jobs + 1):
        for j in range(1, num_jobs + 1):
            if i != j:
                for t in range(processing_times[i], T - processing_times[j] + 1):
                    objective += weights[j] * (t + processing_times[j]) * x[i, j, t]
    model.setObjective(objective, 'minimize')

    # 容量约束
    model.addCons(quicksum(x[0, j, 0] for j in range(num_jobs + 1)) == num_machines, 'Capacity')

    # 分配约束
    for j in range(1, num_jobs + 1):
        model.addCons(quicksum(x[i, j, t] for i in range(num_jobs + 1) if i != j \
                               for t in range(processing_times[i], T - processing_times[j] + 1)) == 1, f'Assign_{j}')

    # 流平衡约束
    for i in range(1, num_jobs + 1):
        for t in range(T - processing_times[i] + 1):
            model.addCons(quicksum(x[j, i, t] for j in range(num_jobs + 1) if j != i and t - processing_times[j] >= 0) \
                          - quicksum(x[i, j, t + processing_times[i]] for j in range(num_jobs + 1) \
                          if j != i and t + processing_times[i] + processing_times[j] <= T) == 0, f'FlowConservation_{i}_{t}')

    # 空闲约束
    for t in range(T):
        model.addCons(quicksum(x[j, 0, t] for j in range(num_jobs + 1) if t - processing_times[j] >= 0)\
                      - quicksum(x[0, j, t + 1] for j in range(num_jobs + 1) if t + processing_times[j] + 1 <= T) == 0, f'Idle_{t}')

    # 模型求解
    model.optimize()
    solution = model.getBestSol()
    schedule = {}
    optimal_alpha = None

    non_zero_vars = {}
    for (i, j, t) in x:
        if model.getVal(x[i, j, t]) != 0:
            non_zero_vars[f'x_{i}_{j}_{t}'] = model.getVal(x[i, j, t])

    for var, val in non_zero_vars.items():
        print(f"{var}: {val}")

    return non_zero_vars, schedule, model.getObjVal()

# 示例调用
processing_times = [6, 4, 3, 6, 5, 6, 4, 8]
weights = [1, 1, 1, 1, 1, 1, 1, 1]
num_machines = 2
solution, schedule, optimal_alpha = ATIF_Solving(processing_times, weights, num_machines, relaxation=True)
print('Hello world')


