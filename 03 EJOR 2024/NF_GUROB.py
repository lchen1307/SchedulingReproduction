import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt


# %% 1 绘制甘特图

def draw_gantt_chart(schedule, start_times, processing_times):
    """
    绘制甘特图，横轴表示时间，纵轴表示机台，使用不同颜色区分工件

    Parameters
    ----------
    schedule (list): 调度顺序
    start_times (dict): 工件的开始时间
    processing_times (dict): 工件的加工时间
    """
    fig, gnt = plt.subplots(figsize=(10, 3))

    # 设置甘特图的X轴范围
    gnt.set_xlim(0, max(start_times[i] + processing_times[i] for i in start_times) + 5)
    gnt.set_xlabel('Time')
    gnt.set_ylabel('Machine')

    # 设置Y轴，只需要一行表示单机调度
    gnt.set_ylim(0, 2)
    gnt.set_yticks([1])
    # gnt.set_yticklabels(['Machine 1'])

    # 为每个工件选择一个不同的颜色
    colors = plt.cm.get_cmap('tab20', len(start_times))

    # 绘制每个工件的条形图，区分不同颜色
    for i, job in enumerate(start_times):
        gnt.broken_barh([(start_times[job], processing_times[job])], (0.5, 1),
                        facecolors=(colors(i)),
                        edgecolor='black',
                        label=f'Job {job}')

    # 添加图例
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

    # 显示甘特图
    plt.tight_layout()
    # plt.savefig("scheduling_image.png", dpi=300)  # 保存为高清png图片
    plt.show()


# %% 2 Single MIP

# def single_machine_MIP_basic(processing_times, setup_times):
#     """
#     最基本的单机调度MIP模型，考虑序列相关设置时间，最小化最大完工时间

#     Parameters
#     ----------
#     processing_times (dict): 每个工件的加工时间，例如 {1:2, 2:4, 3:3}
#     setup_times (dict): 工件间的设置时间，例如 {(1, 2): 3, (1, 3): 2}

#     Returns
#     -------
#     tuple: 包含最优最大完工时间 (makespan) 和调度顺序的元组

#     """

#     # 提取工件集合
#     jobs = list(processing_times.keys())

#     # 创建Gurobi数学模型
#     m = gp.Model("Single_Machine_Scheduling_MIP_Basic")

#     # 两个决策变量：加工顺序和加工时间
#     f = m.addVars(setup_times.keys(), vtype=GRB.BINARY, name='f')
#     C = m.addVars(jobs, vtype=GRB.CONTINUOUS, name='c')
#     C_max = m.addVar(vtype=GRB.CONTINUOUS, name='C_max')

#     # 目标函数：最小化最大完工时间
#     m.setObjective(C_max, GRB.MINIMIZE)

#     # 约束条件
#     # 匹配问题的基本逻辑约束
#     for j in jobs:
#         m.addConstr(gp.quicksum(f[i, j] for i in jobs if (i, j) in setup_times) == 1, name=f"job_{j}_logic_1")

#     for i in jobs:
#         m.addConstr(gp.quicksum(f[i, j] for j in jobs if (i, j) in setup_times) == 1, name=f'job_{i}_logic_2')

#     # 完工时间约束 (C_j >= C_i + p_j + s_ij)
#     for i in jobs:
#         for j in jobs:
#             if i != j:
#                 m.addConstr(C[j] >= C[i] + processing_times[i] + setup_times[i, j] * f[i, j],
#                             name = f'completion_time_{i}_{j}')

#     # 起始工件的完工时间
#     for j in jobs:
#         m.addConstr(C[j] >= processing_times[j], name=f'start_completion_time_{j}')

#     # 定义C_max为最大完工时间
#     for j in jobs:
#         m.addConstr(C_max >= C[j], name=f'Cmax_definition_{j}')

#     # 优化模型并输出结果
#     m.optimize()

#     if m.status == GRB.OPTIMAL:
#         optimal_C_max = C_max.X
#         schedule = []

#         for (i, j) in setup_times:
#             if f[i, j].X > 0.5:
#                 schedule.append((i, j))

#         return optimal_C_max, schedule

#     else:
#         return None, None


# %% 2 Advanced MIP

def single_machine_MIP_advanced(processing_times, setup_times, draw_gantt=False):
    """
    使用MIP求解单机调度问题，考虑序列相关的设置时间，最小化最大完工时间，并可以选择是否绘制甘特图

    Parameters
    ----------
    processing_times (dict): 每个工件的加工时间，例如 {1:2, 2:4, 3:3}
    setup_times (dict): 工件间的设置时间，例如 {(1, 2): 3, (1, 3): 2}
    draw_gantt (bool): 是否绘制甘特图，默认不绘制

    Returns
    -------
    tuple: 包含最优最大完工时间 (makespan)、调度顺序、工件开始时间和完工时间的字典
    """

    jobs = list(processing_times.keys())
    n = len(jobs)

    # 创建Gurobi数学模型
    m = gp.Model("Single_Machine_Scheduling_MIP_Advanced")

    # 决策变量
    x = m.addVars(jobs, range(n), vtype=GRB.BINARY, name="x")
    y = m.addVars(jobs, jobs, range(n - 1), vtype=GRB.BINARY, name="y")
    S = m.addVars(range(n), vtype=GRB.CONTINUOUS, name="S")  # 开始时间
    C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max")  # 最大完工时间

    # 目标函数：最小化最大完工时间
    m.setObjective(C_max, GRB.MINIMIZE)

    # 约束 1：每个位置只能分配一个工件
    for r in range(n):
        m.addConstr(gp.quicksum(x[i, r] for i in jobs) == 1, name=f"position_{r}_unique")

    # 约束 2：每个工件只能分配到一个位置
    for i in jobs:
        m.addConstr(gp.quicksum(x[i, r] for r in range(n)) == 1, name=f"job_{i}_unique")

    # 约束 3：工件的完工时间，包括加工时间和设置时间
    for r in range(n - 1):
        m.addConstr(S[r + 1] >= S[r] + gp.quicksum(processing_times[i] * x[i, r] for i in jobs) +
                    gp.quicksum(setup_times[i, j] * y[i, j, r] for i in jobs for j in jobs if i != j),
                    name=f"completion_time_{r}")

    # 约束 4：逻辑约束，确定顺序切换
    for r in range(n - 1):
        for i in jobs:
            for j in jobs:
                if i != j:
                    m.addConstr(x[i, r] + x[j, r + 1] - 1 <= y[i, j, r], name=f"logic_{i}_{j}_{r}")
                    m.addConstr(y[i, j, r] <= x[i, r], name=f"y_logic_1_{i}_{j}_{r}")
                    m.addConstr(y[i, j, r] <= x[j, r + 1], name=f"y_logic_2_{i}_{j}_{r}")

    # 约束 5：定义最大完工时间 C_max
    m.addConstr(C_max >= S[n - 1] + gp.quicksum(processing_times[i] * x[i, n - 1] for i in jobs),
                name="Cmax_definition")

    # 优化模型
    m.optimize()

    if m.status == GRB.OPTIMAL:
        optimal_C_max = C_max.X
        schedule = []
        start_times = {}  # 保存工件的开始时间
        end_times = {}  # 保存工件的完工时间

        # 输出每个工件的调度顺序和对应的开始时间
        for r in range(n):
            for i in jobs:
                if x[i, r].X > 0.5:
                    schedule.append(i)
                    start_times[i] = S[r].X
                    end_times[i] = S[r].X + processing_times[i]

        # 如果需要绘制甘特图
        if draw_gantt:
            draw_gantt_chart(schedule, start_times, processing_times)

        return optimal_C_max, schedule, start_times, end_times

    else:
        return None, None, None, None


# %% 3 Network Flow

def solve_single_machine_network_flow(processing_times, setup_times, T):
    """
    使用Gurobi求解简化版的网络流问题，不考虑release dates且忽略s_{0j}

    Parameters
    ----------
    processing_times (dict): 每个工件的加工时间，例如 {1:2, 2:4, 3:3}
    setup_times (dict): 工件间的设置时间，例如 {(1, 2): 5, (1, 3): 1}
    T (int): 机器的总时间窗口

    Returns
    -------
    tuple: 包含最优最大完工时间、调度方案、工件开始时间和结束时间的字典
    """

    jobs = list(processing_times.keys())

    # 创建Gurobi模型
    m = gp.Model("Network_Flow_Simplified")

    # 定义变量
    # 定义A1: 从一个工件到另一个工件的调度关系 (i, t) -> (j, t + s_{ij} + p_j)
    # arcs = []
    # for i in jobs:
    #     for j in jobs:
    #         if i != j:
    #             min_setup_time_j = min([setup_times[k, l] for k in jobs for l in jobs if l != j and k != l])
    #             for t in range(min_setup_time_j + processing_times[i], T - setup_times[i, j] - processing_times[j] + 1):
    #                 arcs.append(((i, t), (j, t + setup_times[i, j] + processing_times[j])))

    arcs = []
    for i in jobs:
        for j in jobs:
            if i != j:
                for t in range(processing_times[i], T - setup_times[i, j] - processing_times[j] + 1):
                    arcs.append(((i, t), (j, t + setup_times[i, j] + processing_times[j])))

    # 定义A2: 起点弧，从(0, t)到工件 j
    start_arcs = []
    for j in jobs:
        start_arcs.append(((0, 0), (j, processing_times[j])))

    # 定义A3: 终点弧，从工件 (j, t) 到终点 (0, T)，(j, t) 属于集合 R
    end_arcs = []
    for j in jobs:
        min_setup_time_j = min([setup_times[k, l] for k in jobs for l in jobs if l != j and k != l])
        for t in range(min_setup_time_j + processing_times[j], T + 1):  # t 从最小设置时间 + 加工时间开始到 T
            end_arcs.append(((j, t), (0, T)))

    # 定义所有弧
    all_arcs = arcs + start_arcs + end_arcs

    # 创建流量变量
    x = m.addVars(all_arcs, vtype=GRB.BINARY, name="x")

    # 定义目标函数：最小化最大完工时间 α
    alpha = m.addVar(vtype=GRB.CONTINUOUS, name="alpha")
    m.setObjective(alpha, GRB.MINIMIZE)

    # 约束1: 每个工件只能完成一次 √
    for j in jobs:
        Aj = [a for a in all_arcs if a[1][0] == j]
        m.addConstr(gp.quicksum(x[a] for a in Aj) == 1, name=f"job_completion_{j}")

    # 约束2: 起点到工件的流量必须为1 √
    m.addConstr(gp.quicksum(x[a] for a in start_arcs) == 1, name="start_flow")

    # 约束3: 流平衡约束
    for v in [(i, t) for i in jobs for t in range(T)]:
        m.addConstr(
            gp.quicksum(x[a] for a in all_arcs if a[1] == v) - gp.quicksum(x[a] for a in all_arcs if a[0] == v) == 0,
            name=f"flow_balance_{v}")

    # 约束4: 定义最大完工时间 α 的约束，结合不同弧集合的权重
    for j in jobs:
        Aj = [a for a in all_arcs if a[1][0] == j]

        # A1: 权重为弧的第二个端点的时间 t (即 a[1][1])
        total_cost = gp.quicksum(
            a[1][1] * x[a]
            for a in Aj if a[0][0] != 0
        )

        # A2: 起点弧，使用终点的时间 t
        total_cost += gp.quicksum(
            a[1][1] * x[a]
            for a in Aj if a[0][0] == 0
        )

        # A3: 终点弧的权重为 0
        total_cost += gp.quicksum(0 * x[a] for a in Aj if a[1][0] == 0)

        # 添加约束 α >= ∑ c_a * x_a，确保最大完工时间
        m.addConstr(alpha >= total_cost, name=f"max_completion_time_{j}")

    # 优化模型
    m.optimize()

    m.write('total_model.lp')

    if m.status == GRB.OPTIMAL:
        optimal_alpha = alpha.X
        schedule = []
        start_times = {}  # 保存每个工件的开始时间
        end_times = {}  # 保存每个工件的结束时间

        # 输出调度方案，提取每个工件的开始和结束时间
        for a in all_arcs:
            if x[a].X > 0.5:
                if a[0][0] == 0:  # 起点到第一个工件
                    start_times[a[1][0]] = a[1][1] - processing_times[a[1][0]]
                elif a[1][0] == 0:  # 最后一个工件到终点
                    end_times[a[0][0]] = a[0][1]
                else:  # 工件之间的流动
                    schedule.append(a)

        return optimal_alpha, schedule, start_times, end_times


    elif m.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS...")
        # 计算不可行约束集
        m.computeIIS()
        # 将不可行性报告输出到文件
        m.write("model.ilp")
        return None, None, None, None


# %% 示例算例
# %% 1 验证两个模型的等价性
# %%% 1.1 overleaf上展示的算例

processing_times = {1: 2, 2: 4, 3: 3}
setup_times = {
    (1, 2): 5, (1, 3): 1, (2, 1): 5, (2, 3): 2,
    (3, 1): 1, (3, 2): 2
}

T = 16

# network flow
optimal_alpha, optimal_schedule, start_times, end_times = solve_single_machine_network_flow(processing_times,
                                                                                            setup_times, T)

if optimal_alpha is not None:
    print(f"Optimal C_max (Alpha): {optimal_alpha}")
    print("Optimal schedule:")
    for arc in optimal_schedule:
        print(f"From {arc[0]} to {arc[1]}")
    print("Start times:", start_times)
    print("End times:", end_times)
else:
    print("No optimal solution found.")

# MIP
optimal_C_max, optimal_schedule, start_times, end_times = single_machine_MIP_advanced(processing_times, setup_times)

if optimal_C_max is not None:
    print(f"Optimal C_max: {optimal_C_max}")
    schedule_str = " -> ".join([f"Job {job}" for job in optimal_schedule])
    print(f"Optimal schedule: {schedule_str}")

else:
    print("No optimal solution found.")

# %%% 1.2 另一个简单的算例
processing_times = {1: 2, 2: 4, 3: 3, 4: 6, 5: 5}
setup_times = {
    (1, 2): 3, (1, 3): 2, (1, 4): 4, (1, 5): 3,
    (2, 1): 3, (2, 3): 1, (2, 4): 5, (2, 5): 2,
    (3, 1): 2, (3, 2): 4, (3, 4): 3, (3, 5): 2,
    (4, 1): 4, (4, 2): 3, (4, 3): 2, (4, 5): 5,
    (5, 1): 3, (5, 2): 2, (5, 3): 4, (5, 4): 1,
}

T = 38

# network flow
optimal_alpha, optimal_schedule, start_times, end_times = solve_single_machine_network_flow(processing_times,
                                                                                            setup_times, T)

if optimal_alpha is not None:
    print(f"Optimal C_max (Alpha): {optimal_alpha}")
    print("Optimal schedule:")
    for arc in optimal_schedule:
        print(f"From {arc[0]} to {arc[1]}")
    print("Start times:", start_times)
    print("End times:", end_times)
else:
    print("No optimal solution found.")

# MIP
optimal_C_max, optimal_schedule, _, _ = single_machine_MIP_advanced(processing_times, setup_times, draw_gantt=False)

if optimal_C_max is not None:
    print(f"Optimal C_max: {optimal_C_max}")
    schedule_str = " -> ".join([f"Job {job}" for job in optimal_schedule])
    print(f"Optimal schedule: {schedule_str}")

else:
    print("No optimal solution found.")


# Placeholder for the MIP and Network Flow functions with optimized timing

def single_machine_MIP_advanced(processing_times, setup_times, draw_gantt=False):
    """ Simulating MIP advanced function optimization time """
    jobs = list(processing_times.keys())
    n = len(jobs)

    # Creating Gurobi model
    m = gp.Model("Single_Machine_Scheduling_MIP_Advanced")

    # Variables
    x = m.addVars(jobs, range(n), vtype=GRB.BINARY, name="x")
    y = m.addVars(jobs, jobs, range(n - 1), vtype=GRB.BINARY, name="y")
    S = m.addVars(range(n), vtype=GRB.CONTINUOUS, name="S")
    C_max = m.addVar(vtype=GRB.CONTINUOUS, name="C_max")

    # Objective function
    m.setObjective(C_max, GRB.MINIMIZE)

    # Constraints
    for r in range(n):
        m.addConstr(gp.quicksum(x[i, r] for i in jobs) == 1)
    for i in jobs:
        m.addConstr(gp.quicksum(x[i, r] for r in range(n)) == 1)
    for r in range(n - 1):
        m.addConstr(S[r + 1] >= S[r] + gp.quicksum(processing_times[i] * x[i, r] for i in jobs) +
                    gp.quicksum(setup_times[i, j] * y[i, j, r] for i in jobs for j in jobs if i != j))
    for r in range(n - 1):
        for i in jobs:
            for j in jobs:
                if i != j:
                    m.addConstr(x[i, r] + x[j, r + 1] - 1 <= y[i, j, r])
                    m.addConstr(y[i, j, r] <= x[i, r])
                    m.addConstr(y[i, j, r] <= x[j, r + 1])
    m.addConstr(C_max >= S[n - 1] + gp.quicksum(processing_times[i] * x[i, n - 1] for i in jobs))

    # Measure optimization time
    start_time = time.time()
    m.optimize()
    optimize_time = time.time() - start_time

    return optimize_time, None, None, None


def solve_single_machine_network_flow(processing_times, setup_times, T):
    """ Simulating Network Flow function optimization time """
    jobs = list(processing_times.keys())

    # Creating Gurobi model
    m = gp.Model("Network_Flow_Simplified")

    # Variables
    arcs = []
    for i in jobs:
        for j in jobs:
            if i != j:
                for t in range(processing_times[i], T - setup_times[i, j] - processing_times[j] + 1):
                    arcs.append(((i, t), (j, t + setup_times[i, j] + processing_times[j])))

    start_arcs = [((0, 0), (j, processing_times[j])) for j in jobs]
    end_arcs = [((j, t), (0, T)) for j in jobs for t in range(processing_times[j], T + 1)]

    all_arcs = arcs + start_arcs + end_arcs

    x = m.addVars(all_arcs, vtype=GRB.BINARY, name="x")
    alpha = m.addVar(vtype=GRB.CONTINUOUS, name="alpha")
    m.setObjective(alpha, GRB.MINIMIZE)

    for j in jobs:
        Aj = [a for a in all_arcs if a[1][0] == j]
        m.addConstr(gp.quicksum(x[a] for a in Aj) == 1)
    m.addConstr(gp.quicksum(x[a] for a in start_arcs) == 1)
    for v in [(i, t) for i in jobs for t in range(T)]:
        m.addConstr(
            gp.quicksum(x[a] for a in all_arcs if a[1] == v) - gp.quicksum(x[a] for a in all_arcs if a[0] == v) == 0)

    for j in jobs:
        Aj = [a for a in all_arcs if a[1][0] == j]
        total_cost = gp.quicksum(a[1][1] * x[a] for a in Aj if a[0][0] != 0)
        total_cost += gp.quicksum(a[1][1] * x[a] for a in Aj if a[0][0] == 0)
        m.addConstr(alpha >= total_cost)

    # Measure optimization time
    start_time = time.time()
    m.optimize()
    optimize_time = time.time() - start_time

    return optimize_time, None, None, None


# Compare optimization times
def compare_optimization_runtimes():
    job_sizes = [5, 8, 10]
    num_instances = 5
    result_data = []

    for num_jobs in job_sizes:
        for _ in range(num_instances):
            processing_times, setup_times = generate_random_instance(num_jobs)

            # Measure time for MIP optimization
            mip_runtime, _, _, _ = single_machine_MIP_advanced(processing_times, setup_times)

            # Measure time for Network Flow optimization
            T = sum(processing_times.values()) + sum(setup_times.values()) // len(setup_times)
            flow_runtime, _, _, _ = solve_single_machine_network_flow(processing_times, setup_times, T)

            result_data.append({
                "Job Size": num_jobs,
                "MIP Optimization Time (s)": mip_runtime,
                "Network Flow Optimization Time (s)": flow_runtime
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(result_data)
    return df


# Generate and display the comparison results
df_optimization_runtimes = compare_optimization_runtimes()

import ace_tools as tools;

tools.display_dataframe_to_user(name="Optimization Runtime Comparison", dataframe=df_optimization_runtimes)