"""
@filename: DWATIF.py
@author: Long Chen
@time: 2025-01-14
"""

from __future__ import division, print_function
from DataClass import DataReader
import networkx as nx
from gurobipy import *
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

        # initialize parameters
        self.Iter = 0
        self.dual_convexCons = 1
        self.dual_CapacityCons = []
        self.dual_AssignmentCons = []
        self.SP_totalCost = []

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

    def NetworkShortestPath(self, arc_reduced_cost = None):

        # 创建网络图、寻找最短路
        G = nx.DiGraph
        if arc_reduced_cost is None:
            arc_reduced_cost = {}

        for i in range(self.NumJob + 1):
            for j in range(self.NumJob + 1):
                for t in self.var_x[i][j].keys():
                    node = (i, j, t)
                    G.add_node(node)

                    if (i == 0) & (j == 0):
                        G.add_edge(node, node, weight = arc_reduced_cost.get((i, j, t), 1))
                    else:
                        next_node = (j, i, t + 1)
                        G.add_edge(node, next_node, weight = arc_reduced_cost.get((i, j, t), 1))

        self.graph = G

        start_node = (0, 0, 0)
        end_node = (0, 0, self.T)
        shortest_path_length = nx.dijkstra_path_length(G, source=start_node, target=end_node)

        return shortest_path_length

    def initializeModel(self):
        # 主问题的初始化没什么问题
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
        self.AssignmentCons.append(AssignmentCons_temp)

        # initialize the convex combination constraints
        self.convexCons = self.RDWM.addConstr(1 * self.var_Artificial == 1, name = 'convex cons')

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

        def update_subproblem_objective(self, dual_pi, dual_wj):
            """更新定价子问题的目标函数"""
            # 获取弧时间索引变量
            var_x = self.var_x

            # 初始化目标函数
            objective = LinExpr()

            # 遍历所有可能的弧 (i, j) t
            for i in range(self.NumJob + 1):
                for j in range(self.NumJob + 1):
                    if i == j:
                        continue
                    for t in var_x[i][j]:
                        var = var_x[i][j][t]

                        # 假设目标函数是基于 j 的完成时间和权重
                        completion_time = t + self.ProcessingTime[j]
                        cost = self.Weight[j] * completion_time  # 完成成本

                        # 计算影子成本
                        # 假设 dual_pi 是容量约束的对偶变量，但在这里可能不适用
                        # 主要的对偶变量应该是任务分配约束的对偶变量
                        # 对于任务 j 的分配，假设每个任务 j 的对偶变量是 dual_wj[j]
                        reduced_cost = cost - dual_wj[j]  # 假设每个任务 j 有一个对偶变量

                        # 更新目标函数
                        objective += reduced_cost * var

            # 设置目标函数
            self.subProblem.setObjective(objective, GRB.MINIMIZE)

        def optimizePhase_1(self):
            # 初始化参数
            self.dual_CapacityCons.append([0.0])
            self.dual_AssignmentCons = []  # 存储任务分配约束的对偶变量

            # 添加人工变量的约束
            self.CapacityCons.append(self.RDWM.addConstr(self.var_Artificial == self.NumMachine, name='Capacity Cons'))

            # 添加任务分配约束
            for j in range(1, self.NumJob + 1):
                AssignmentCons_j = []
                for _ in range(1):  # 只有一个约束，因此只添加一次
                    # 此处应根据论文中的约束形式正确添加
                    pass
                self.AssignmentCons.append(AssignmentCons_j)

            # 优化主问题
            self.RDWM.setObjective(self.var_Artificial, GRB.MINIMIZE)
            self.RDWM.optimize()

            # 获取对偶变量
            dual_pi = self.RDWM.getAttr(GRB.Attr.Pi, self.CapacityCons)
            dual_wj = self.RDWM.getAttr(GRB.Attr.Pi, self.AssignmentCons)

            # 更新子问题的目标函数
            self.update_subproblem_objective(dual_pi[0][0], dual_wj.flatten().tolist())

            # 优化子问题
            self.subProblem.optimize()

            # 输出结果
            print("Subproblem Objective Value:", self.subProblem.objVal)
            for i in range(self.NumJob + 1):
                for j in range(self.NumJob + 1):
                    for t in self.var_x[i][j]:
                        var = self.var_x[i][j][t]
                        if var.x > 0:
                            print(f"x[{i}, {j}, {t}] = {var.x}")


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

            if self.subProblem.objval >= -1e-6:
                print('No new column will be generated, coz no negative reduced cost columns')
                break
            else:
                self.Iter = self.Iter + 1

                # complete the total cost of subproblem solutions
                # the total cost is the coefficient of RMP when new column is added
                # totalCost_subProblem = sum(self.Weight[j] * self.var_x[j] for j in range(self.T))
                # self.SP_totalCost.append(totalCost_subProblem)

                # update constraints in RDWM
                col = Column()
                for j in range(self.NumJob):
                    col.addTerms(sum(self.var_x[j].x))

                col.addTerms(1, self.convexCons)

                self.var_lambda.append(self.RDWM.addVar(lb = 0.0
                                                       , ub = GRB.INFINITY
                                                       , obj = 0.0
                                                       , vtype = GRB.CONTINUOUS
                                                       , name = 'lam_phase_1_' + str(self.Iter)
                                                       , column = col))
                self.RDWM.optimize()

                # update dual variables
                if self.RDWM.objval <= -1e-6:
                    print('--- obj of phase 1 reaches 0, phase 1 ends ---')
                    break
                else:
                    for i in range(self.NumJob):
                        self.dual_AssignmentCons = self.AssignmentCons[j].pi
                    self.dual_CapacityCons = self.convexCons.pi

                # reset objective for subproblem in phase 1
                obj_sub_phase_1 = - quicksum(self.dual_AssignmentCons[j] - self.var_x[j] \
                                             for j in range(self.NumJob)) - self.dual_convexCons

                self.subProblem.setObjective(obj_sub_phase_1, GRB.MINIMIZE)



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
    PMATIF_instance.ProcessingTime = [0, 6, 4, 3, 6, 5]
    PMATIF_instance.Weight = [0, 1, 1, 1, 1, 1]
    PMATIF_instance.DueDate = [0, 20, 18, 17, 16, 15, 14, 13]
    PMATIF_instance.T = sum(sorted(PMATIF_instance.ProcessingTime, reverse=True)[
                            :math.ceil(PMATIF_instance.NumJob / PMATIF_instance.NumMachine)])
    PMATIF_instance.solvePMATIF()

