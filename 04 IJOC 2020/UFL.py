import numpy as np

# 问题数据示例
n_facilities = 3  # 设施数量
m_customers = 2  # 客户数量
# 每个设施的开设成本
f = [3, 2, 4]
# 分配成本c[j][i]表示设施j对客户i的分配成本
c = [
    [1, 2],
    [3, 1],
    [2, 2]
]

# 拉格朗日松弛参数
max_iter = 1000  # 最大迭代次数
alpha = 2  # 初始步长因子
tol = 1e-5  # 收敛容忍度

# 初始化拉格朗日乘子
lambda_i = np.zeros(m_customers)

best_LB = -np.inf  # 最佳下界
best_UB = np.inf  # 最佳上界
best_y = None  # 最佳设施开设方案
best_assignment = None  # 最佳客户分配

for iter in range(max_iter):
    # 1. 求解拉格朗日松弛子问题
    y = np.zeros(n_facilities, dtype=int)
    x = np.zeros((n_facilities, m_customers), dtype=int)


    # 计算每个设施的总效益
    facility_benefits = np.zeros(n_facilities)
    for j in range(n_facilities):
        benefit = f[j]
        for i in range(m_customers):
            benefit += min(c[j][i] + lambda_i[i], 0)
        facility_benefits[j] = benefit
        if benefit < 0:
            y[j] = 1
            # 确定客户分配
            for i in range(m_customers):
                if c[j][i] + lambda_i[i] < 0:
                    x[j][i] = 1

    # 计算下界
    LB = sum(facility_benefits[j] for j in range(n_facilities) if y[j] == 1) - sum(lambda_i)
    if LB > best_LB:
        best_LB = LB

    # 2. 启发式构造可行解
    opened_facilities = [j for j in range(n_facilities) if y[j] == 1]

    # 如果没有设施被选中，强制选择开设成本最小的设施
    if not opened_facilities:
        j_min = np.argmin(f)
        opened_facilities = [j_min]
        # 修复：更新y变量以反映强制开启的设施
        y[j_min] = 1  # <--- 新增此行
        heuristic_cost = f[j_min] + sum(c[j_min][i] for i in range(m_customers))
    else:
        # 计算每个客户分配到最近设施的成本
        assignment = []
        assignment_cost = 0
        for i in range(m_customers):
            min_cost = np.inf
            for j in opened_facilities:
                if c[j][i] < min_cost:
                    min_cost = c[j][i]
            assignment_cost += min_cost
        heuristic_cost = sum(f[j] for j in opened_facilities) + assignment_cost

    # 更新最佳上界
    if heuristic_cost < best_UB:
        best_UB = heuristic_cost
        best_y = y.copy()
        # 记录具体分配方案
        best_assignment = []
        for i in range(m_customers):
            min_cost = np.inf
            best_j = -1
            for j in opened_facilities:
                if c[j][i] < min_cost:
                    min_cost = c[j][i]
                    best_j = j
            best_assignment.append(best_j)

    # 3. 计算次梯度
    gradient = np.zeros(m_customers)
    for i in range(m_customers):
        gradient[i] = sum(x[j][i] for j in range(n_facilities)) - 1

    # 4. 更新步长和乘子
    grad_norm_sq = np.sum(gradient ** 2)
    if grad_norm_sq > 0:
        step_size = alpha * (best_UB - LB) / grad_norm_sq
    else:
        step_size = 0

    lambda_i += step_size * gradient

    # 减小步长因子
    alpha *= 0.95

    # 显示迭代信息
    if iter % 100 == 0:
        print(f"Iteration {iter}: LB = {best_LB:.2f}, UB = {best_UB:.2f}, Gap = {best_UB - best_LB:.2f}")

    # 收敛检查
    if best_UB - best_LB < tol:
        print(f"Converged at iteration {iter}!")
        break

# 输出最终结果
print("\nOptimal Solution:")
print("Opened facilities:", [j for j in range(n_facilities) if best_y[j]])
print("Customer assignments:", best_assignment)
print(f"Total cost: {best_UB}")