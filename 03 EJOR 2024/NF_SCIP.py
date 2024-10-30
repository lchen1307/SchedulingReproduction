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

# 定义 s_ij_bar
s_ij_bar = [min([setup_time[i][j] for i in range(jobs + 1) if i != j]) for j in range(jobs)]

# 定义节点集 V
R = [(j + 1, t + s_ij_bar[j] + release_dates[j] + processing_times[j]) for j in range(jobs) \
     for t in range(T + 1 - (s_ij_bar[j] + release_dates[j] + processing_times[j]))]
O = [(0, t) for t in range(T + 1)]
V = R + O

# 定义弧集 A
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

# 建立模型
model = Model('Single Machine Scheduling Problem')

# 决策变量
arc_vars = {a: model.addVar(vtype='B', name=f'x_{a[0][0]} + {a[0][1]} + {a[1][0]} + {a[1][1]}')
            for a in A1 + A2 + A3 + A4}

# 目标变量
alpha = model.addVar(vtype='C', name='alpha')
model.setObjective(alpha)

# 约束1：每个工件只能安排一次
for j in range(1, jobs + 1):
    A_j_edges = [((i, t), (j, t_next)) for ((i, t), (j, t_next)) in (A1 + A2 + A3) if i != j]
    model.addCons(quicksum(arc_vars[(i, t), (j, t_next)] for ((i, t), (j, t_next)) in A_j_edges) == 1)

# 约束2：dummy job 到第一个作业的约束
model.addCons(quicksum(arc_vars[(0, t), (j, t_next)] for ((start, t), (j, t_next)) in A2 if start == 0) == 1)

# 约束3：流平衡约束
for v in V:
    in_edges = [((i, t), v) for ((i, t), (j, t_next)) in A1 + A2 + A3 + A4 if (j, t_next) == v]
    out_edges = [(v, (j, t_next)) for ((i, t), (j, t_next)) in A1 + A2 + A3 + A4 if (i, t) == v]
    model.addCons(quicksum(arc_vars[edge] for edge in in_edges) ==
                  quicksum(arc_vars[edge] for edge in out_edges))

# 约束4：makespan约束
for (j, t) in R:
    model.addCons(alpha >= t, f"makespan_{j}_{t}")

# 求解
model.writeProblem("model.lp", "", True, True)
model.conflictAnalyze()
result = model.optimize()

if model.getStatus() == "optimal":
    print("Optimal Solution Found!")
    for var in arc_vars.values():
        if model.getVal(var) > 0.5:
            print(f"{var.name} = {model.getVal(var)}")
    print(f"Alpha (makespan): {model.getVal(alpha)}")
else:
    print("No Optimal Solution Found.")