"""
@filename: MIP_SCIP.py
@author: Long Chen
@time: 2024-10-30
"""

from pyscipopt import Model, quicksum

T = 25
jobs = 5
release_dates = [1, 3, 10, 2, 4]
processing_times = [1, 2, 1, 2, 2]
setup_time = [
    [2, 5, 1, 5, 7],
    [0, 9, 8, 1, 5],
    [4, 0, 6, 5, 3],
    [1, 3, 0, 8, 2],
    [10, 2, 3, 0, 7],
    [8, 1, 5, 7, 0]
]

# 定义四类边
s_ij_bar = [None] * jobs
for j in range(jobs):
    s_ij_values = [setup_time[i][j] for i in range(jobs + 1) if i != (j + 1)]
    s_ij_bar[j] = min(s_ij_values)

R = [(j + 1, t + s_ij_bar[j] + release_dates[j] + processing_times[j]) for j in range(jobs) \
     for t in range(T + 1 - (s_ij_bar[j] + release_dates[j] + processing_times[j]))]
O = [(0, t) for t in range(T + 1)]
V = R + O

A1 = [((i, t), (j, t + setup_time[i][j - 1] + processing_times[j - 1])) \
      for (i, t) in R
      for (j, t_next) in R
      if release_dates[j - 1] <= t < t_next == t + setup_time[i][j - 1] + processing_times[j - 1]
      and (i != j)]
A2 = [((0, t), (j + 1, t + setup_time[0][j] + processing_times[j]))
      for t in range(T) for j in range(jobs)
      if t + setup_time[0][j] + processing_times[j] <= T]
A3 = [((j, t), (0, T)) for (j, t) in R]
A4 = [((j, t), (j, t + 1)) for (j, t) in V if (j, t + 1) in V]


# 建立模型并调用SCIP求解
model = Model('Single Machine Scheduling Problem')

# 决策变量
arc_vars = {a: model.addVar(vtype='B', name=f'x_{a[0][0]} + {a[0][1]} + {a[1][0]} + {a[1][1]}')
            for a in A1 + A2 + A3 + A4}

# 目标函数
alpha = model.addVar(vtype = 'C', name = 'alpha')
model.setObjective(alpha)

# 约束条件1：所有工件都只需被加工一次
for j in range(1, jobs + 1):
    A_j_edges = [((i, t), (j, t_next)) for ((i, t), (j, t_next)) in (A1 + A2 + A3)\
                 if t_next == t + setup_time[i][j-1] + processing_times[j-1] and i != j]
    model.addCons(quicksum(arc_vars[(i, t), (j, t_next)] for ((i, t), (j, t_next)) in A_j_edges) == 1)

# 约束条件2：dummy job -> first job
model.addCons(quicksum(arc_vars[(0, t), (j, t_next)] for ((start, t), (j, t_next)) in A2
                       if start == 0) == 1)

# 约束条件3：流平衡约束
for v in V:
    in_edges = [((i, t), v) for ((i, t), (j, t_next)) in A1 + A2 + A3 + A4 if (j, t_next) == v]
    out_edges = [(v, (j, t_next)) for ((i, t), (j, t_next)) in A1 + A2 + A3 + A4 if (i, t) == v]
    model.addCons(quicksum(arc_vars[edge] for edge in in_edges) ==
                  quicksum(arc_vars[edge] for edge in out_edges))

# 约束条件4：makespan
for (j, t) in R:
    model.addCons(alpha >= t * quicksum(arc_vars[((i, t), (j, t_next))] for ((i, t), (j, t_next)) in A3))

# 模型求解并输出结果
model.writeProblem("model.lp", "", True, True)
result = model.optimize()

if model.getStatus() == "optimal":
    print("Optimal Solution Found!")
    for var in arc_vars.values():
        if model.getVal(var) > 0.5:
            print(f"{var.name} = {model.getVal(var)}")
    print(f"Alpha (makespan): {model.getVal(alpha)}")
else:
    print("No Optimal Solution Found.")