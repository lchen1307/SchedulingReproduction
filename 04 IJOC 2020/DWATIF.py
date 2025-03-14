"""
@filename: DWATIF.py
@author: Long Chen
@time: 2025-01-14
"""

from __future__ import division, print_function
from collections import defaultdict
from DataClass import DataReader
from gurobipy import *
import networkx as nx
import math

class PMATIF:

    def __init__(self):
        # initialize data
        self.NumMachine = 0
        self.NumJob = 0
        self.T = 0
        self.ProcessingTime = []
        self.Weight = []
        self.DueDate = []

        # initialize variables and constraints in RMP
        self.CapacityCons = []
        self.AssignmentCons = []
        self.var_lambda = []

        # initialize variables in subproblems
        self.var_x = []

        # initialize the network graph
        self.graph = []
        self.Paths = {}

        # initialize parameters
        self.Iter = 0
        self.dual_convexCons = 1
        self.dual_CapacityCons = []
        self.dual_AssignmentCons = []
        self.SP_totalCost = []
        self.reduced_dual = [] # 最短路问题权重的中间变量参数

    def readData(self, path, job_count, machine_count):
        data_groups = DataReader.readData(path, job_count)

        # 暂时只需要一组数据进行测试，后续可以进行拓展
        if data_groups:
            first_group = data_groups[0]
            jobs = first_group.jobs

            self.NumJob = len(jobs)
            self.NumMachine = machine_count
            self.ProcessingTime = [job.processing_time for job in jobs]
            self.ProcessingTime = [0] + self.ProcessingTime
            self.T = sum(sorted(self.ProcessingTime, reverse=True)[:math.ceil(self.NumJob / self.NumMachine)])
            self.Weight = [job.weight for job in jobs]
            self.Weight = [0] + self.Weight
            self.DueDate = [job.due_date for job in jobs]
            self.DueDate = [0] + self.DueDate

    def GreedyRule(self):

        sorted_indices = sorted(range(self.NumJob), key=lambda x: self.Weight[x] / self.ProcessingTime[x], reverse=True)
        # sorted_indices = sorted(range(self.NumJob), key = lambda x: self.ProcessingTime[x]) # SPT 最短加工时间优先规则
        # sorted_indices = sorted(range(self.NumJob), key=lambda x: self.DueDate[x]) # EDD 最早交货期优先规则
        # sorted_indices = sorted(range(self.NumJob), key=lambda x: -self.Weight[x]) # 最大权重优先规则

        machine_completion_times = [0] * self.NumMachine
        assignment = [[] for _ in range(self.NumMachine)]

        for job_idx in sorted_indices:
            selected_machine = min(range(self.NumMachine), key = lambda m: machine_completion_times[m])
            assignment[selected_machine].append(job_idx)
            machine_completion_times[selected_machine] += self.ProcessingTime[job_idx]

        # 计算加权总完工时间
        total_weighted_completion_time = 0
        start_times = []
        for m in range(self.NumMachine):
            start_time = 0
            machine_start_times = []
            for job_idx in assignment[m]:
                machine_start_times.append((job_idx, start_time))
                start_time += self.ProcessingTime[job_idx]
            start_times.append(machine_start_times)

        for m in range(self.NumMachine):
            lambda_key = f"lambda_{m+1}"  # 机器编号从1开始
            path = []
            # 在路径开头添加0
            prev_job = 0
            for job_idx, start_time in start_times[m]:
                # 作业编号从1开始
                current_job = job_idx + 1
                path.append(f"x_{prev_job}_{current_job}_{start_time}")
                prev_job = current_job
            # 在路径结尾添加0，开始时间为机器的完成时间
            path.append(f"x_{prev_job}_0_{machine_completion_times[m]}")
            self.Paths[lambda_key] = path

            assignment_output = []
            for m in range(self.NumMachine):
                # 机器编号从1开始，作业编号从1开始
                assignment_output.append([job_idx + 1 for job_idx in assignment[m]])


    def NetworkShortestPath(self):

        # 创建网络图、寻找最短路
        G = nx.DiGraph()

        for i in range(self.NumJob + 1):
            for t in range(self.T + 1):
                node = (i, t)
                G.add_node(node)

        for i in range(self.NumJob + 1):
            for j in range(self.NumJob + 1):
                for t in self.var_x[i][j].keys():
                    arc_reduced_cost = self.Weight[j] * (t + self.ProcessingTime[j])
                    G.add_edge((i, t - self.ProcessingTime[i]), (j, t), weight = arc_reduced_cost)

        self.graph = G

        start_node = (0, 0)
        end_node = (0, self.T)
        shortest_path_length = nx.dijkstra_path_length(G, source=start_node, target=end_node)
        shortest_path = nx.dijkstra_path(G, source=start_node, target=end_node)

        arcs = []
        for i in range(len(shortest_path) - 1):
            start = shortest_path[i]
            end = shortest_path[i + 1]
            arc = f'x_{start[0]}_{end[0]}_{end[1]}'
            arcs.append(arc)

        key = f'lam_phase_1_{self.Iter}'
        self.Paths[key] = arcs

        return shortest_path_length, shortest_path

    def YenKShortestPaths(self, k):

        # 创建网络图
        G = nx.DiGraph()

        for i in range(self.NumJob + 1):
            for t in range(self.T + 1):
                node = (i, t)
                G.add_node(node)

        for i in range(self.NumJob + 1):
            for j in range(self.NumJob + 1):
                for t in self.var_x[i][j].keys():
                    arc_reduced_cost = self.Weight[j] * (t + self.ProcessingTime[j])
                    G.add_edge((i, t - self.ProcessingTime[i]), (j, t), weight=arc_reduced_cost)

        self.graph = G

        start_node = (0, 0)
        end_node = (0, self.T)

        paths = list(nx.shortest_simple_paths(G, start_node, end_node, weight='weight'))

        if len(paths) < k:
            k = len(paths)

        for idx in range(k):
            path = paths[idx]

            arcs = []
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                arc = f'x_{start[0]}_{end[0]}_{end[1]}'
                arcs.append(arc)

            ##################################

            key = f'lambda_{idx+3}'
            self.Paths[key] = arcs

    def updatePaths(self):
        for path_key in list(self.Paths.keys()):
            arcs = self.Paths[path_key]
            last_arc = arcs[-1]
            parts = last_arc.split('_')
            if len(parts) != 4 or parts[0] != 'x':
                print(f"Invalid format for arc: {last_arc}")
                continue
            last_time = int(parts[3])

            if last_time < self.T:
                new_arcs = []
                for t in range(last_time + 1, self.T + 1):
                    new_arc = f"x_0_0_{t}"
                    new_arcs.append(new_arc)

                self.Paths[path_key] = arcs + new_arcs

    def updateReducedDual(self):
        self.reduced_dual = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

        for l, constr in enumerate(self.RDWM.getConstrs()):
            expr = self.RDWM.getRow(constr)
            # 遍历线性表达式的每一项
            for i in range(expr.size()):
                var = expr.getVar(i)  # 获取第 i 个变量
                coeff = expr.getCoeff(i)  # 获取第 i 个系数（如果需要）
                var_name = var.VarName
                if var_name.startswith('lambda_'):
                    lambda_index = int(var_name.split('_')[1])
                    if f'lambda_{lambda_index}' in self.Paths:
                        x_list = self.Paths[f'lambda_{lambda_index}']
                        for x in x_list:
                            x_parts = x.split('_')
                            if len(x_parts) == 4 and x_parts[0] == 'x':
                                i = int(x_parts[1])
                                j = int(x_parts[2])
                                t = int(x_parts[3])
                                self.reduced_dual[i][j][t][l] = 1

# 帮我写一个函数def updateReducedDual(self)来计算reduced_dual并将其更新到self.reduced_dual中。具体计算要求为：
# reduced_dual有四个索引，分别是i, j, t, l. i和j是从0到self.NumJob，t的索引可能是任意的整数（最大不超过self.T）,
# l的索引取决于数学模型的约束数。首先，进入到数学模型self.RDWM中，每个约束条件代表一个索引l，第一个约束索引为0. 然后，
# 对于每个约束，其具体形式为：lambda_1 + lambda_2 + lambda_3 + lambda_4 + lambda_5
#    + lambda_6 + lambda_7 = 2，需要提取每个决策变量。对于每个决策变量，如lambda_1，需要在self.Paths这个字典中
#    找到key为lambda_1的value列表，列表是形式为['x_0_1_0', 'x_0_2_0', 'x_0_3_0']，然后把列表中每个元素的数字
#    提取出来，在这个例子中，由于有x_0_1_0，表示x_i_j_t，那么更新到reduced_dual中就是reduced_dual[0][1][0][1]
#    为1（reduced_dual[i][j][t][l]）。值得注意的是，对于这个l, 对应的每个lambda中的每个x都需要遍历计算，只要存在
#    对应的x就是1，不存在就是0

    def initializeModel(self):

        self.GreedyRule()

        self.ProcessingTime = [0] + self.ProcessingTime
        self.DueDate = [0] + self.DueDate
        self.Weight = [0] + self.Weight

        # initialize master problem and subproblem
        self.RDWM = Model('RDWM')
        self.subProblem = Model('subProblem')

        # close log information
        self.RDWM.setParam('OutputFlag', 0)
        self.subProblem.setParam('OutputFlag', 0)

        # add initial artificial variable in RDWMP, in order to start the algorithm
        self.var_Artificial = self.RDWM.addVar(lb = 0.0
                                            , ub = GRB.INFINITY
                                            , obj = 0.0
                                            , vtype = GRB.CONTINUOUS
                                            , name = 'Artificial')

        # add temp capacity and assignment constraint, in order to start the algorithm
        self.CapacityCons.append(self.RDWM.addConstr(-self.var_Artificial == self.NumMachine, name = 'Capacity Cons'))

        AssignmentCons_temp = []
        for i in range(self.NumJob):
            AssignmentCons_temp.append(self.RDWM.addConstr(-self.var_Artificial == 1, name = 'Assignment Cons'))
        self.AssignmentCons.extend(AssignmentCons_temp)

        # initialize the convex combination constraints
        self.convexCons = self.RDWM.addConstr(1 * self.var_Artificial == 1, name = 'Convex Cons')

        self.RDWM.write('initial_RDWM.lp')

        # 初始化 self.var_x 为一个嵌套列表，其中每个元素是一个字典
        self.var_x = [[{} for _ in range(self.NumJob + 1)] for _ in range(self.NumJob + 1)]

        # 定义变量
        for i in range(self.NumJob + 1):
            for j in range(self.NumJob + 1):
                if (i == 0) & (j == 0):
                    for t in range(0, self.T + 1):
                        self.var_x[i][j][t] = self.subProblem.addVar(
                            lb=0.0,
                            ub=GRB.INFINITY,
                            obj=0.0,
                            vtype=GRB.CONTINUOUS,
                            name=f'x_{i}_{j}_{t}'
                        )
                elif i != j:
                    for t in range(self.ProcessingTime[i], self.T - self.ProcessingTime[j] + 1):
                        self.var_x[i][j][t] = self.subProblem.addVar(
                            lb=0.0,
                            ub=GRB.INFINITY,
                            obj=0.0,
                            vtype=GRB.CONTINUOUS,
                            name=f'x_{i}_{j}_{t}')

        for j in range(self.NumJob + 1):
            self.var_x[0][j][0] = self.subProblem.addVar(
                lb=0.0,
                ub=GRB.INFINITY,
                obj=0.0,
                vtype=GRB.CONTINUOUS,
                name=f'x_{0}_{j}_{0}')

        # add flow conservation constraint to the subproblem
        for i in range(1, self.NumJob + 1):
            for t in range(self.T - self.ProcessingTime[i] + 1):
                self.subProblem.addConstr(quicksum(self.var_x[j][i][t] for j in range(self.NumJob + 1) if j != i and t - self.ProcessingTime[j] >= 0) \
                          - quicksum(self.var_x[i][j][t + self.ProcessingTime[i]] for j in range(self.NumJob + 1) \
                          if j != i and t + self.ProcessingTime[i] + self.ProcessingTime[j] <= self.T) == 0, f'FlowConservation_{i}_{t}')

        # add idleness conservation constraint to the subproblem
        for t in range(self.T):
            self.subProblem.addConstr(quicksum(self.var_x[j][0][t] for j in range(self.NumJob + 1) if t - self.ProcessingTime[j] >= 0) \
                          - quicksum(self.var_x[0][j][t + 1] for j in range(self.NumJob + 1) if t + self.ProcessingTime[j] + 1 <= self.T) == 0, f'Idleness_{t}')

        self.RDWM.write('initial_ATIFRDWM.lp')
        self.subProblem.write('initial_ATIFsubProblem.lp')



    def optimizePhase_1(self):
        # initialize parameter
        self.dual_CapacityCons.append([0.0])
        for j in range(self.NumJob):
            self.dual_AssignmentCons.append([0.0])

        obj_master_phase_1 = self.var_Artificial
        obj_sub_phase_1 = -quicksum(
            self.dual_CapacityCons[0] * self.var_x[0][j][0]
            for j in range(self.NumJob)
            if 0 in self.var_x[0] and j in self.var_x[0][0] and 0 in self.var_x[0][0][j]) - self.dual_convexCons

        # set objective for RMP of Phase 1
        self.RDWM.setObjective(obj_master_phase_1, GRB.MINIMIZE)

        # set objective for subproblem of Phase 1
        self.subProblem.setObjective(obj_sub_phase_1, GRB.MINIMIZE)
        self.subProblem.write('initial_ATIFsubProblem_2.lp')

        # in order to make initial model is feasible, we set initial convex constraints to Null,
        # and in later iteration, we set the RHS of convex constraint to 1
        self.RDWM.chgCoeff(self.convexCons, self.var_Artificial, 0.0)

        # Phase 1 of Danzig-Wolfe decomposition: to ensure the initial model is feasible
        print('---------- start Phase 1 optimization ----------')\

        while True:
            print('Iter: ', self.Iter)
            self.RDWM.write('DWATIF_master.lp')
            self.subProblem.optimize()
            # shortest_path_length, shortest_path = self.NetworkShortestPath()
            self.YenKShortestPaths(5)
            self.updatePaths()

            if self.subProblem.ObjVal >= -1e-6:
                print('No new column will be generated, coz no negative reduced cost columns')
            else:
                self.Iter = self.Iter + 1

                # complete the total cost of subproblem solutions
                # the total cost is the coefficient of RMP when new column is added
                # totalCost_subProblem = sum(self.Weight[j] * self.var_x[j] for j in range(self.T))
                # self.SP_totalCost.append(totalCost_subProblem)

                # update constraints in RDWM
                # 遍历 self.Paths 中的所有路径
                for path_key, arcs in self.Paths.items():
                    # 创建一个新的列
                    col = Column()

                    # 计算每个约束的系数
                    for j in range(self.NumJob):
                        count = 0
                        for arc in arcs:
                            parts = arc.split('_')
                            if len(parts) == 4 and parts[0] == 'x':
                                current_j = int(parts[2])
                                if current_j == (j + 1):
                                    count += 1
                        col.addTerms(count, self.AssignmentCons[j])

                    # 计算容量约束的系数
                    count = 0
                    for arc in arcs:
                        parts = arc.split('_')
                        if len(parts) == 4 and parts[0] == 'x':
                            if parts[1] == '0' and parts[3] == '0':
                                count += 1
                    col.addTerms([count], self.CapacityCons)

                    # 添加凸组合约束的系数
                    col.addTerms(1.0, self.convexCons)

                    # 目标函数的系数
                    total_cost = 0
                    for arc in arcs:
                        parts = arc.split('_')
                        if len(parts) == 4 and parts[0] == 'x':
                            current_j = int(parts[2])
                            current_time = int(parts[3])

                        total_cost += self.Weight[current_j] * (current_time + self.ProcessingTime[current_j])

                    # 创建一个新的变量并添加到主问题中
                    new_var = self.RDWM.addVar(
                        lb=0.0,
                        ub=GRB.INFINITY,
                        obj=total_cost,
                        vtype=GRB.CONTINUOUS,
                        name=f'{path_key}',
                        column=col
                    )
                    self.var_lambda.append(new_var)

                artificial_var = None
                for var in self.RDWM.getVars():
                    if var.VarName == "Artificial":
                        artificial_var = var
                        break

                if artificial_var is not None:
                    # 删除目标函数中与 Artificial 相关的部分
                    self.RDWM.remove(artificial_var)
                    self.RDWM.update()

                # 删除约束条件 Convenx_Cons
                convex_cons = None
                for constr in self.RDWM.getConstrs():
                    if constr.ConstrName == "Convex_Cons":
                        convex_cons = constr
                        break

                if convex_cons is not None:
                    self.RDWM.remove(convex_cons)
                    self.RDWM.update()

                self.RDWM.optimize()
                self.updateReducedDual()

                # update dual variables
                if self.RDWM.ObjVal <= -1e-6:
                    print('--- obj of phase 1 reaches 0, phase 1 ends ---')
                    break
                else:
                    for i in range(self.NumJob):
                        self.dual_AssignmentCons = self.AssignmentCons[j].pi
                    self.dual_CapacityCons = self.convexCons.pi
                #
                # # reset objective for subproblem in phase 1
                # obj_sub_phase_1 = - quicksum(self.dual_AssignmentCons[j] - self.var_x[j] \
                #                              for j in range(self.NumJob)) - self.dual_convexCons
                #
                # self.subProblem.setObjective(obj_sub_phase_1, GRB.MINIMIZE)



    def solvePMATIF(self):
        # initialize the RMP and subproblem
        self.initializeModel()

        # Dantzig-Wolfe decomposition
        print('-------------------------------------------------------')
        print('---------- start Dantzig-Wolfe decomposition ----------')
        print('-------------------------------------------------------')

        self.optimizePhase_1()



if __name__ == '__main__' :
    PMATIF_instance = PMATIF()
    # PMATIF_instance.readData(r'../Dataset/wt40.txt', job_count = 40, machine_count = 2)
    PMATIF_instance.NumJob = 5
    PMATIF_instance.NumMachine = 2
    PMATIF_instance.ProcessingTime = [6, 4, 3, 6, 5]
    PMATIF_instance.Weight = [1, 1, 1, 1, 1]
    PMATIF_instance.DueDate = [20, 18, 17, 16, 15, 14, 13]
    PMATIF_instance.T = sum(sorted(PMATIF_instance.ProcessingTime, reverse=True)[
                            :math.ceil(PMATIF_instance.NumJob / PMATIF_instance.NumMachine)])
    PMATIF_instance.solvePMATIF()


# 3.11在完成self.Paths的更新后，新建一个函数，完成两件事
#   ①对于正常计算加工时间，需要把最后一个节点延长到(0,T)
#   ②对于最短路算法计算的路径，最后的节点需要向前推加工时间
# 3.11另一个更重要的工作：检查向主问题添加列的系数计算是否正确！！！大概率主问题不可行的原因就出在这里
# 3.12要做的事情：学习子问题的目标函数是如何更新的，然后把这个逻辑纳入到代码中