"""

@filename: B&B.py
@author: Long Chen
@time: 2025-01-20

"""

from NodeClass import Node
from DataClass import DataReader
import numpy as np
import random
import copy
import math

# Branch-cut-and-price framework
def Branch_and_Cut(ATIF_model, x_var, summary_interval):
    Relax_ATIF_model = ATIF_model.relax()
    Relax_ATIF_model.optimize()
    global_UB = np.inf
    global_LB = Relax_ATIF_model.Objval
    eps = 1e-6

    incumbent_node = None
    Gap = np.inf
    feasible_sol_cnt = 0
    Cuts_pool = {}
    Cuts_LHS = {}
    Cut_cnt = 0

    '''
        Branch and Cut starts
    '''

    # create initial node
    Queue = []
    node = Node()
    node.local_LB = global_LB
    node.local_UB = np.inf
    node.model = Relax_ATIF_model.copy()
    node.model.setParam('OutputFlag', 0)
    node.cnt = 0
    Queue.append(node)
    cnt = 0
    Global_UB_change = []
    Global_LB_change = []

    while (len(Queue) > 0 and global_UB - global_LB > eps):
        # select the current node
        current_node = Queue.pop()
        cnt += 1

        # solve the current model
        current_node.model.optimize()
        Solution_status = current_node.model.Status

        '''
        OPTIMAL = 2
        INFEASIBLE = 3
        UNBOUNDED = 5
        '''

        # check whether the current solution is integer and execute prune step

        '''
            is_integer: mark whether the current solution is integer solution
            Is_Pruned: mark whether the current solution is pruned 
        '''

        is_integer = True
        Is_Pruned = False

        if (Solution_status == 2):
            for var in current_node.model.getVars():
                if (var.VarName.startswith('x')):
                    current_node.x_sol[var.VarName] = var.x

                    # record the branchable variable
                    if (abs(round(var.x, 0) - var.x) >= eps):
                        is_integer = False

            '''
                Cuts Generation
            '''

            # cut generation
            temp_cut_cnt = 0
            data = DataReader.readData()
            if (is_integer == False):
                customer_set = list(range(data.nodeNum))
                while (temp_cut_cnt <= 5):
                    sample_num = random.choice(customer_set[3:])
                    selected_customer_set = random.sample(customer_set, sample_num)
                    estimated_veh_num = 0
                    total_demand = 0
                    for customer_set in selected_customer_set:
                        total_demand += data.demand(customer_set)
                    estimated_veh_num = math.ceil(total_demand / data.capacity)
                    current_node.model_vars = x_var

                    # create cut
                    cut_lhs = LinExpr(0)
                    for key in x_var.keys():
                        key_org = key[0]
                        key_des = key[1]
                        key_vehicle = key[2]
                        if (key_org not in selected_customer_set and key_des in selected_customer_set):
                            var_name = 'x_' + str(key_org) + '_' + str(key_des) + '_' + str(key_vehicle)
                            cut_lhs.addTerms(1, current_node.model.getVarName(var_name))
                    Cut_cnt += 1
                    temp_cut_cnt += 1
                    cut_name = 'Cut_' + str(Cut_cnt)
                    Cuts_pool[cut_name] = current_node.model.addConstr(cut_lhs >= estimated_veh_num, name = cut_name)
                    Cuts_LHS[cut_name] = cut_lhs

                # lazy update
                current_node.model.update()

            '''
                Cuts Generation ends
            '''

            current_node.model.optimize()
            is_integer = True

            if (current_node.model.status == 2):
                for var in current_node.model.getVars():
                    if (var.VarName.startswith('x')):
                        current_node.x_sol[var.VarName] = copy.deepcopy(current_node.model.getVarByName(var.VarName).x)
                        if (abs(round(var.x, 0) - var.x) >= eps):
                            is_integer = False
                            current_node.branch_var_list.append(var.VarName)

            else:
                continue

            if (is_integer == True):
                feasible_sol_cnt += 1
                current_node.is_integer = True
                current_node.local_LB = current_node.model.ObjVal
                current_node.local_UB = current_node.model.ObjVal

                # if the solution is integer, update the UB of global and update the incumbent
                if (current_node.local_UB < global_UB):
                    global_UB = current_node.local_UB
                    incumbent_node = Node.deepcopy_node(current_node)

            if (is_integer == False):

                # For integer solution node, update the LB and UB also
                current_node.is_integer = False
                current_node.local_UB = global_UB
                current_node.local_LB = current_node.model.ObjVal

            '''
                PRUNE step
            '''

            # prune by optimality
            if (is_integer == True):
                Is_Pruned = True

            # prune by bound
            if (is_integer == False and current_node.local_LB > global_UB):
                Is_Pruned = True

            Gap = round(100 * (global_UB - global_LB) / global_LB, 2)

        elif (Solution_status != 2):
            # the current node is infeasible or unbounded
            is_integer = False

            '''
                PRUNE step
            '''

            # prune by infeasibility
            Is_Pruned = True

            continue

        '''
            BRANCH step
        '''

        if (Is_Pruned == False):
            # select the branch variable: choose the value which is cloest to 0.5
            branch_var_name = None
            min_diff = 100

            for var_name in current_node.branch_var_list:
                if (abs(current_node.x_sol[var_name] - 0.5) < min_diff):
                    branch_var_name = var_name
                    min_diff = abs(current_node.x_sol[var_name] - 0.5)

            if (cnt % summary_interval == 0):
                print('Branch var name: ', branch_var_name, '\t, branch var value: ', current_node.x_sol[branch_var_name])

            left_var_bound = (int)(current_node.x_sol)[branch_var_name]
            right_var_bound = (int)(current_node.x_sol)[branch_var_name] + 1

            # create two child nodes
            left_node = Node.deepcopy_node(current_node)
            right_node = Node.deepcopy_node(current_node)

            # create left child node
            temp_var = left_node.model.getVarByName(branch_var_name)
            left_node.model.addConstr(temp_var <= left_var_bound, name = 'branch_left_' + str(cnt))
            left_node.model.setParam('OutputFlag', 0)
            left_node.model.update()
            cnt += 1
            left_node.cnt = cnt

            # create right child node
            temp_var = left_node.model.getVarByName(branch_var_name)
            right_node.model.addConstr(temp_var >= right_var_bound, name = 'branch_right_' + str(cnt))
            right_node.model.setParam('OutputFlag', 0)
            right_node.model.update()
            cnt += 1
            left_node.cnt = cnt
            Queue.append(left_node)
            Queue.append(right_node)

            # update the global LB, explore all the left node
            temp_global_LB = np.inf

            for node in Queue:
                node.model.optimize()

                if (node.model.status == 2):
                    if (node.model.ObjVal <= temp_global_LB and node.model.ObjVal <= global_UB):
                        temp_global_LB = node.model.ObjVal

            global_LB = temp_global_LB
            Global_UB_change.append(global_UB)
            Global_LB_change.append(global_LB)

        if (cnt % summary_interval == 0):
            print('\n\n=====================')
            print('Queue length: ', len(Queue))
            print('\n ---------- \n', cnt, 'UB = ', global_UB, ' LB = ', global_LB, '\t Gap = ', Gap, ' %', 'feasible_sol_cnt: ', feasible_sol_cnt)
            print('Cut pool size: ', len(Cuts_pool))
            NAME = list(Cuts_LHS.keys())[-1]
            print('RHS: ', estimated_veh_num)
            print('Cons Num: ', current_node.model.NumConstrs)

    # all the nodes are explored, update the LB and UB
    incumbent_node.model.optimize()
    global_UB = incumbent_node.model.ObjVal
    global_LB = global_UB
    Gap = round(100 * (global_UB - global_LB) / global_LB, 2)
    Global_UB_change.append(global_UB)
    Global_LB_change.append(global_LB)

    print('\n\n\n\n')
    print('-------------------------------------------')
    print('          Branch and Cut terminates        ')
    print('           Optimal solution found          ')
    print('-------------------------------------------')
    print('\nIter cnt = ', cnt, ' \n\n')
    print('\nFinal Gap = ', Gap, '% \n\n')
    print(' ------ Optimal Solution ------')
    for key in incumbent_node.x_sol.keys():
        if (incumbent_node.x_sol[key] > 0):
            print(key, ' = ', incumbent_node.x_sol[key])

    print('\nOptimal Obj: ', global_LB)

    return incumbent_node, Gap, Global_UB_change, Global_LB_change


# branch and cut solve the IP model
incumbent_node, Gap, Global_UB_change, Global_LB_change = Branch_and_Cut(model, x_var, summary_interval = 100)

for key in incumbent_node.x_sol.keys():
    if (incumbent_node.x_sol[key] > 0):
        print(key, ' = ', incumbent_node.x_sol[key])
